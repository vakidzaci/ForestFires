def cleaner(image, **kwargs):
    autoencoder = load_model("./flask_celery/saved_model_denoiseAE_SP_NOISE_AND_BACKGROUND")
    img_og = image
    img_og = np.zeros((image.shape[0], image.shape[1], 3))

    """Add the channels to the needed image one by one"""
    img_og [:,:,0] = image
    img_og [:,:,1] = image
    img_og [:,:,2] = image

    if img_og is not None:

        width, height = img_og.shape[1], img_og.shape[0]
        INPUT_H = 420
        INPUT_W = 540
        out_img = np.full(img_og.shape, 255)
        n_w = (width // INPUT_W)
        n_h = (height // INPUT_H)
        s = 1000
        #     print(n_w,n_h)
        for w in range(n_w + 1):
            for h in range(n_h + 1):
                img = np.full((INPUT_H, INPUT_W,img_og.shape[2]), 0)
                start_h = h * INPUT_H
                end_h = h * INPUT_H + INPUT_H
                start_w = w * INPUT_W
                end_w = w * INPUT_W + INPUT_W
                if end_h > height:
                    end_h = height
                if end_w > width:
                    end_w = width
                target = img_og[start_h:end_h, start_w:end_w]

                img[:target.shape[0], :target.shape[1], :] = target
                img = np.asarray(img, dtype="float32")
                img = cv2.resize(img, (540, 420))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img / 255.0
                img = np.reshape(img, (420, 540, 1))
                # print(img.shape)

                decoded_imgs = autoencoder(np.array([img])).numpy()
                img = decoded_imgs[0]
                #         print(img.shape)
                img *= 255

                out_shape = out_img[start_h:end_h, start_w:end_w].shape
                out_img[start_h:end_h, start_w:end_w] = img[:out_shape[0], :out_shape[1], :]
        return out_img
