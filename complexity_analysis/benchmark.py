import time
import torch
from state_encoder import MultiModalEncoder
from FusionMamba.fusion_module import FusionMamba as TokenFusionMamba
from FusionMamba.fusionmamba import FusionMamba as ImageFusionMamba
from PPO_agent import ActorCritic

def count_parameters(model):
    """
    Counts learnable parameters in a PyTorch module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_complexity_benchmark(config):
    """
    Performs computational complexity analysis:
    - Counts parameter size of each model component
    - Estimates MACs / FLOPs (using thop if available)
    - Measures average forward pass latency
    """
    device = torch.device(config.training.device)
    
    # 1. Initialize models
    encoder = MultiModalEncoder(
        visual_shape=config.encoder.visual_input_shape,
        proprio_dim=config.encoder.proprioceptive_dim,
        embed_dim=config.encoder.embed_dim
    ).to(device)
    
    fusion_mamba = TokenFusionMamba(
        d_model=config.mamba.d_model,
        d_state=config.mamba.d_state,
        d_conv=config.mamba.d_conv,
        expand=config.mamba.expand,
        num_layers=config.mamba.num_layers,
        dropout=config.mamba.dropout
    ).to(device)
    
    in_channels = getattr(config.mamba, 'in_channels', 1)
    base_channels = getattr(config.mamba, 'base_channels', 32)
    out_channels = getattr(config.mamba, 'out_channels', 1)
    image_fusion_model = ImageFusionMamba(
        in_channels=in_channels,
        base_channels=base_channels,
        out_channels=out_channels
    ).to(device)

    actor_critic = ActorCritic(
        state_dim=config.mamba.d_model,
        action_dim=config.agent.action_dim,
        hidden_dim=config.agent.hidden_dim,
        action_space_type=config.agent.action_space_type
    ).to(device)
    
    # 2. Count parameters
    encoder_params = count_parameters(encoder)
    mamba_token_params = count_parameters(fusion_mamba)
    image_fusion_params = count_parameters(image_fusion_model)
    ac_params = count_parameters(actor_critic)
    rl_framework_params = encoder_params + mamba_token_params + ac_params
    
    print("-" * 55)
    print("Parameter Complexity Analysis:")
    print(f"  - 2D FusionMamba Image Model: {image_fusion_params:,} parameters")
    print(f"  - MultiModalEncoder:          {encoder_params:,} parameters")
    print(f"  - Token FusionMamba Module:   {mamba_token_params:,} parameters")
    print(f"  - ActorCritic Policy:         {ac_params:,} parameters")
    print(f"  - RL Framework Total:         {rl_framework_params:,} parameters")
    print("-" * 55)
    
    # 3. Compute FLOPs using thop if installed
    dummy_vis = torch.randn(1, *config.encoder.visual_input_shape, device=device)
    dummy_proprio = torch.randn(1, config.encoder.proprioceptive_dim, device=device)
    
    try:
        from thop import profile
        print("Calculating FLOPs using THOP package...")
        
        # Profile state encoder
        enc_flops, _ = profile(encoder, inputs=(dummy_vis, dummy_proprio), verbose=False)
        
        # Profile FusionMamba
        dummy_vis_feats = torch.randn(1, config.encoder.embed_dim, device=device)
        dummy_proprio_feats = torch.randn(1, config.encoder.embed_dim, device=device)
        mamba_flops, _ = profile(fusion_mamba, inputs=(dummy_vis_feats, dummy_proprio_feats), verbose=False)
        
        # Profile ActorCritic
        dummy_fused = torch.randn(1, config.mamba.d_model, device=device)
        ac_flops, _ = profile(actor_critic, inputs=(dummy_fused,), verbose=False)
        
        print(f"FLOPs / MACs estimation (Single sample batch):")
        print(f"  - MultiModalEncoder FLOPs: {enc_flops / 1e6:.2f} MFLOPs")
        print(f"  - FusionMamba FLOPs:       {mamba_flops / 1e6:.2f} MFLOPs")
        print(f"  - ActorCritic FLOPs:      {ac_flops / 1e6:.2f} MFLOPs")
        print(f"  - Total Forward FLOPs:    {(enc_flops + mamba_flops + ac_flops) / 1e6:.2f} MFLOPs")
        
    except ImportError:
        print("THOP package not installed. Run 'pip install thop' to display exact FLOP complexity.")
        
    print("-" * 50)
    
    # 4. Measure latency
    print("Measuring forward pass execution latency...")
    warmup_steps = 10
    measure_steps = 100
    
    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            feats = encoder(dummy_vis, dummy_proprio)
            fused = fusion_mamba(feats["visual"], feats["proprio"])
            _ = actor_critic(fused)
            
    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(measure_steps):
        with torch.no_grad():
            feats = encoder(dummy_vis, dummy_proprio)
            fused = fusion_mamba(feats["visual"], feats["proprio"])
            _ = actor_critic(fused)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / measure_steps) * 1000
    print(f"Average Forward Latency: {avg_latency_ms:.3f} ms on {device.type.upper()}")
    print("-" * 50)
