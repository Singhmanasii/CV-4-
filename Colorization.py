import numpy as np
import cv2

proto_file = 'C:/Users/singh/Downloads/colorizing/colorizing/model/colorization_deploy_v2.prototxt'
model_file = 'C:/Users/singh/Downloads/colorizing/colorizing/model/colorization_release_v2.caffemodel'
hull_pts = 'C:/Users/singh/Downloads/colorizing/colorizing/model/pts_in_hull.npy'
img_path = 'C:/Users/singh/Downloads/colorizing/colorizing/images/nature.jpg'

net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

img = cv2.imread(img_path)
scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

resized = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose([1, 2, 0])
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

L = cv2.split(lab_img)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
colorized = np.clip(colorized, 0, 255)

colorized = cv2.cvtColor(colorized.astype("uint8"), cv2.COLOR_LAB2BGR)
colorized = (255 * colorized).astype("uint8")

print("L channel shape:", L.shape)
print("ab_channel shape:", ab_channel.shape)
print("ab_channel range:", ab_channel.min(), ab_channel.max())
print("colorized range:", colorized.min(), colorized.max())

L_normalized = (L - L.min()) / (L.max() - L.min()) * 255
ab_channel_a = (ab_channel[:, :, 0] - ab_channel[:, :, 0].min()) / (ab_channel[:, :, 0].max() - ab_channel[:, :, 0].min()) * 255
ab_channel_b = (ab_channel[:, :, 1] - ab_channel[:, :, 1].min()) / (ab_channel[:, :, 1].max() - ab_channel[:, :, 1].min()) * 255

cv2.imshow("L channel", L_normalized.astype("uint8"))
cv2.imshow("ab_channel_a", ab_channel_a.astype("uint8"))
cv2.imshow("ab_channel_b", ab_channel_b.astype("uint8"))

img = cv2.resize(img, (250, 500))
colorized = cv2.resize(colorized, (250, 500))
result = cv2.hconcat([img, colorized])

cv2.imshow("Grayscale -> Colour", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
