import torch
from models.transformer_relative import Transformer

def compare_transformers():
    """Compare absolute vs relative positional encodings"""
    
    # Set up parameters
    batch_size = 4
    seq_len = 15  # Make sequence longer to have room for outputs
    input_size = 20
    hidden_size = 64
    output_size = 10
    num_heads = 4
    num_layers = 2
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Parameter explanation:
    # - lengths[i]: position where input ends for batch element i
    # - max_out_length: how many output positions to extract
    # - out_lengths[i]: actual output length (for masking)
    # The model extracts outputs from positions [lengths[i], lengths[i] + max_out_length)
    # So we need: lengths[i] + max_out_length <= seq_len
    
    lengths = torch.tensor([5, 7, 4, 6])  # Leave room for 5 output positions
    out_lengths = torch.tensor([5, 5, 5, 5])
    max_out_length = 5
    
    print("=" * 80)
    print("Transformer Comparison: Absolute vs Relative Positional Encodings")
    print("=" * 80)
    
    # 1. Absolute Positional Encoding (Original)
    print("\n1. ABSOLUTE POSITIONAL ENCODING (Original Implementation)")
    print("-" * 80)
    model_absolute = Transformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        use_relative_positions=False  # Use absolute positions
    )
    
    with torch.no_grad():
        output_abs = model_absolute(x, lengths, out_lengths, max_out_length)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_abs.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_absolute.parameters()):,}")
    
    # 2. Relative Positional Encoding (Transformer-XL)
    print("\n2. RELATIVE POSITIONAL ENCODING (Transformer-XL Style)")
    print("-" * 80)
    model_relative = Transformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        use_relative_positions=True  # Use relative positions
    )
    
    with torch.no_grad():
        output_rel = model_relative(x, lengths, out_lengths, max_out_length)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_rel.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model_relative.parameters()):,}")
    
    # # 3. Key Differences
    # print("\n3. KEY DIFFERENCES")
    # print("-" * 80)
    # print("Absolute Positional Encoding:")
    # print("  - Adds sinusoidal position encodings to input embeddings")
    # print("  - Position information is fixed and added before attention")
    # print("  - Each position has the same encoding regardless of context")
    
    # print("\nRelative Positional Encoding (Transformer-XL):")
    # print("  - Incorporates position information INTO the attention mechanism")
    # print("  - Uses relative distances between positions (i-j) instead of absolute positions")
    # print("  - Includes learnable bias terms (u and v) for content and position")
    # print("  - More flexible for capturing positional relationships")
    
    # # 4. Performance Characteristics
    # print("\n4. EXPECTED BENEFITS OF RELATIVE POSITIONS")
    # print("-" * 80)
    # print("  • Better generalization to longer sequences than seen during training")
    # print("  • More interpretable attention patterns based on relative distances")
    # print("  • Can capture inductive biases about local vs distant relationships")
    # print("  • Often performs better on tasks where relative position matters more")
    
    print("\n" + "=" * 80)
    print("Setup complete! You can now train and compare both models.")
    print("=" * 80)

if __name__ == "__main__":
    compare_transformers()