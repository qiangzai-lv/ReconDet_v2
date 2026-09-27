_base_ = ['./grounding_dino_vggt_pretrain_scannet.py']

model = dict(
    reconstruction_decoder=dict(
        _delete_=True,
        query_dims=512,
        semantic_dims=256,
        spatial_dims=512,
        num_layers=6,
        num_heads=8,
        feedforward_channels=2048,
        num_feature_levels=4,
        num_points=4,
        dropout=0.0),
    freeze_modules=['language_model'])
