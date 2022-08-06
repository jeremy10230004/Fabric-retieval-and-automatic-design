import model
import Tool

if __name__ == "__main__":
    Batch_size = 64
    steps = 30000
    train_data, train_label = Tool.load_img()
    m = model.CWGANdiv(train_data, train_label)
    m.compile(optimizer="RMSProp")
    m.Train(batch_size=Batch_size, steps=steps)
    Tool.sample_images(m, m.latent_dim, m.label_kind)
    m.g.save("./cwgan_g_v1.h5")
    m.d.save("./cwgan_d_v1.h5")
