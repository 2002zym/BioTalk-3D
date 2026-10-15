dataset_config = {
    'EATalker': {
        'dirname': 'EATalker',
        'annot_type': 'inhouse_blendshape_weight',  # D6 风格的 51 维 blendshape
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 8,  # 允许直接索引预训练身份（例如 UniTalker 的 id=4）
        'pca': False,
    },
    # 直接复用 UniTalker D6（中文语音，51 维 ARKit blendshape）
    'D6': {
        'dirname': 'D6_unitalker_Chinese_speech',
        'annot_type': 'inhouse_blendshape_weight',
        'scale': 1.0,
        'annot_dim': 51,
        'subjects': 8,
        'pca': False,
    },
}
