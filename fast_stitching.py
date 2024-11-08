import os
import cv2
import numpy as np

class Stitcher:
    def __init__(self):
        pass

    def check_and_make_dir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def file_setting(self):
        left_img = cv2.imread('./tt/1.jpg')
        base_img = cv2.imread('./tt/2.jpg')
        right_img = cv2.imread('./tt/3.jpg')
        result_dir = './results/stitched_result.jpg'
        
        self.check_and_make_dir('./results/')
        return left_img, base_img, right_img, result_dir

    def remove_black_border(self, img):
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold the image to create a binary mask
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # Find contours in the binary mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour which will be the stitched area
        max_area = 0
        best_rect = (0, 0, img.shape[1], img.shape[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
        x, y, w, h = best_rect
        return img[y:y+h, x:x+w]

    def linearBlending(self, img_left, img_right):
        # Find the dimensions of the final blended image
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        height = max(h1, h2)
        width = max(w1, w2)

        # Create new images with the size of the final image
        img_left_large = np.zeros((height, width, 3), dtype=np.uint8)
        img_right_large = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the images onto the large canvases
        img_left_large[:h1, :w1] = img_left
        img_right_large[:h2, :w2] = img_right

        # Create an overlap mask
        overlap_mask = np.logical_and(
            np.any(img_left_large != 0, axis=2),
            np.any(img_right_large != 0, axis=2)
        )

        # Initialize the alpha mask
        alpha_mask = np.zeros((height, width), dtype=np.float32)

        # Compute the alpha mask for blending
        for i in range(height):
            overlap_indices = np.where(overlap_mask[i])[0]
            if len(overlap_indices) > 0:
                minIdx = overlap_indices[0]
                maxIdx = overlap_indices[-1]
                if maxIdx > minIdx:
                    alpha = np.linspace(1, 0, maxIdx - minIdx + 1)
                    alpha_mask[i, minIdx:maxIdx+1] = alpha

        # Convert alpha_mask to 3 channels
        alpha_mask_3c = np.dstack([alpha_mask]*3)

        # Perform linear blending
        blended = (img_left_large * alpha_mask_3c + img_right_large * (1 - alpha_mask_3c)).astype(np.uint8)

        # Handle non-overlapping regions
        blended[~overlap_mask] = img_left_large[~overlap_mask] + img_right_large[~overlap_mask]

        return blended

    def stitching(self, img_left, img_right, flip=False):
        print('SIFT Feature Detection and Matching...')
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_left, None)
        kp2, des2 = sift.detectAndCompute(img_right, None)

        # Use BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test as per Lowe's paper
        good_matches = []
        src_pts = []
        dst_pts = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                src_pts.append(kp1[m.queryIdx].pt)
                dst_pts.append(kp2[m.trainIdx].pt)

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)

        print('Estimating Homography...')
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        print('Warping Images...')
        # Warp the right image to align with the left image
        height_left, width_left = img_left.shape[:2]
        height_right, width_right = img_right.shape[:2]
        panorama_size = (width_left + width_right, max(height_left, height_right))
        img_right_warped = cv2.warpPerspective(img_right, H, panorama_size)

        # Place the left image onto the panorama canvas
        panorama = np.zeros_like(img_right_warped)
        panorama[0:height_left, 0:width_left] = img_left

        print('Blending Images...')
        blended = self.linearBlending(panorama, img_right_warped)

        if flip:
            blended = cv2.flip(blended, 1)

        print('Cropping Result...')
        cropped_result = self.remove_black_border(blended)

        return cropped_result

if __name__ == '__main__':
    stitcher = Stitcher()
    left_img, base_img, right_img, result_dir = stitcher.file_setting()

    print('Stitching Left and Base Images...')
    img_left = cv2.flip(base_img, 1)
    img_right = cv2.flip(left_img, 1)
    LM_img = stitcher.stitching(img_left, img_right, flip=True)
    cv2.imwrite('./results/intermediate_result.jpg', LM_img)

    print('Stitching Intermediate Result and Right Image...')
    img_left = LM_img
    img_right = right_img
    final_image = stitcher.stitching(img_left, img_right, flip=False)

    cv2.imwrite(result_dir, final_image)
    print('Stitching Completed. Result saved to', result_dir)
