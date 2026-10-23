import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
# from lime import lime_image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input
from skimage.segmentation import mark_boundaries

# -------------------------------------------------
# 1. Create Your Own Model (NO load_model)
# -------------------------------------------------
inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D()(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(3, activation='softmax')(x)  # Example: 3 classes

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Model is untrained – but LIME still works for demonstration
print("Model Created (No Load Model Used)")

# -------------------------------------------------
# 2. Dataset & LIME Setup
# -------------------------------------------------
No_of_Dataset = 3
# explainer = lime_image.LimeImageExplainer()


# -------------------------------------------------
# Prediction function for LIME
# -------------------------------------------------
def predict_fn(images):
    processed_imgs = []

    for im in images:
        im = cv2.resize(im, (224, 224)).astype(np.float32)
        im = preprocess_input(im)
        processed_imgs.append(im)

    processed_imgs = np.array(processed_imgs)
    preds = model.predict(processed_imgs)
    return preds




# # -------------------------------------------------
# # 3. MAIN LOOP (Your Style)
# # -------------------------------------------------
# for n in range(1, No_of_Dataset):
#
#     Images = np.load("Images_" + str(n + 1) + ".npy")[170:175]
#
#     for i in range(len(Images)):
#         img = Images[i]
#         img_org = img.astype(np.uint8)
#
#         img_resized = cv2.resize(img_org, (224, 224))
#
#         # -------------------------------------
#         # 4. Get LIME Explanation
#         # -------------------------------------
#         explanation = explainer.explain_instance(
#             img_resized,
#             predict_fn,
#             top_labels=1,
#             hide_color=0,
#             num_samples=1000
#         )
#
#         # temp, mask = explanation.get_image_and_mask(
#         #     explanation.top_labels[0],
#         #     positive_only=True,
#         #     num_features=5,
#         #     hide_rest=False
#         # )
#         # mask = cv2.threshold(img_resized)
#         ret, mask = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
#
#         # lime_img = mark_boundaries(temp / 255.0, mask)
#         # lime_img = np.uint8(lime_img * 255)
#
#         lime_img = mask
#         lime_img = cv2.resize(lime_img, (img.shape[1], img.shape[0]))
#
#         # -------------------------------------
#         # 5. Save Outputs (Your Style)
#         # -------------------------------------
#         cv2.imwrite(
#             f"./Results/XAI_Image_Results/Dataset_{n + 1}Img{i + 1}_ORG.jpg", img_org)
#
#         cv2.imwrite(
#             f"./Results/XAI_Image_Results/Dataset_{n + 1}Img{i + 1}_LIME.jpg",
#             cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR))
#



import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

No_of_Dataset = 3

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

for n in range(2, No_of_Dataset):
    Images = np.load('Image_' + str(n+1) + '.npy')[170:175]

    for i in range(len(Images)):

        img = Images[i]                        # numpy array (H,W,3)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img.astype(np.uint8))  # FIX ✔

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(img_pil).unsqueeze(0)

        finalconv_name = 'layer4'
        features, gradients = [], []

        def save_features(m, i, o):
            features.append(o)

        def save_gradients(m, gi, go):
            gradients.append(go[0])

        layer = dict([*model.named_modules()])[finalconv_name]
        layer.register_forward_hook(save_features)
        layer.register_backward_hook(save_gradients)

        output = model(input_tensor)
        pred_class = output.argmax().item()

        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0].cpu().detach().numpy()[0]
        fmap = features[0].cpu().detach().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for j, w in enumerate(weights):
            cam += w * fmap[j]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.4 * heatmap + 0.6 * img).astype(np.uint8)
        cv2.imwrite(f"./Results/XAI_Image_Results/Dataset_{n + 1}Img{i + 1}_org.jpg", img)
        cv2.imwrite(f"./Results/XAI_Image_Results/Dataset_{n+1}Img{i+1}_GradCAM.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


