from ultralytics.nn.attention.attention import ParallelPolarizedSelfAttention


def add_attention(model):
    at0 = model.model.model[4]
    n0 = at0.cv2.conv.out_channels
    at0.attention = ParallelPolarizedSelfAttention(n0)

    at1 = model.model.model[6]
    n1 = at1.cv2.conv.out_channels
    at1.attention = ParallelPolarizedSelfAttention(n1)

    at2 = model.model.model[8]
    n2 = at2.cv2.conv.out_channels
    at2.attention = ParallelPolarizedSelfAttention(n2)
    return model