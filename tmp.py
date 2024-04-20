import cv2

def resize_image(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path, -1)
    # 获取图像的原始尺寸
    height, width = img.shape[:2]
    # 计算新的尺寸（每个边长为原来的 1/4）
    new_height, new_width = height // 4, width // 4
    # 使用 cv2.resize() 函数调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height))
    # 保存调整大小后的图像
    cv2.imwrite(output_path, resized_img)

# 使用示例
image_path = '../testImages/test.tif'
output_path = '../testImages/test_samll.tif'
resize_image(image_path, output_path)
