import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def npy_to_colorized_depth_image(npy_file_path, output_image_path, colormap="plasma"):
    """
    Convert a .npy depth file to a colorized depth image and save it using the 'plasma' colormap.

    Args:
        npy_file_path (str): Path to the input .npy depth file.
        output_image_path (str): Path to save the colorized depth image.
        colormap (str): Matplotlib colormap name (default: "plasma").
    """
    # Load the depth map from the .npy file
    depth_map = np.load(npy_file_path)

    # Normalize the depth map to the range [0, 1] for Matplotlib
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Apply the colormap using Matplotlib
    cmap = plt.get_cmap(colormap)
    colorized_depth_map = cmap(depth_map_normalized)  # Returns RGBA values

    # Convert RGBA to RGB and scale to [0, 255]
    colorized_depth_map = (colorized_depth_map[:, :, :3] * 255).astype(np.uint8)

    # Save the colorized depth map as an image
    cv2.imwrite(output_image_path, cv2.cvtColor(colorized_depth_map, cv2.COLOR_RGB2BGR))

    print(f"Colorized depth image saved to {output_image_path}")

def preprocess_images(color_image_path, depth_image_path, target_size=(518, 392)):
    """
    Preprocess and resize color and depth images to a fixed size.
    
    Args:
        color_image_path (str): Path to the color image file.
        depth_image_path (str): Path to the depth image file.
        target_size (tuple): Target size for resizing (width, height).
        
    Returns:
        o3d.geometry.Image: Resized color image.
        o3d.geometry.Image: Resized depth image.
    """
    # Load and resize the color image
    color_image = Image.open(color_image_path).convert("RGB")
    color_image_resized = color_image.resize(target_size, Image.NEAREST)

    # Load and resize the depth image
    # depth_image = Image.open(depth_image_path)
    # depth_image_resized = depth_image.resize(target_size, Image.NEAREST)

    # Convert to NumPy arrays
    color_image_np = np.asarray(color_image_resized, dtype=np.uint8)
    # depth_image_np = np.asarray(depth_image_resized, dtype=np.float32)
    depth_image_np = np.load(depth_image_path)
    depth_image_3 = np.zeros_like(color_image_resized)
    depth_image_3[:,:,0] = depth_image_np
    depth_image_3[:,:,1] = depth_image_np
    depth_image_3[:,:,2] = depth_image_np
    # depth_image_np = depth_image_np[:,:,:3]
    # Convert to Open3D Image format
    color_image_o3d = o3d.geometry.Image(color_image_np)
    depth_image_o3d = o3d.geometry.Image(depth_image_np)

    return color_image_o3d, depth_image_o3d


def depth_to_metric_depth(depth_image, scale=1000.0):
    """
    Convert raw depth image to metric depth (in meters).
    
    Args:
        depth_image (np.ndarray): Raw depth image (H x W).
        scale (float): Scale factor to convert to meters (e.g., 1000 for mm to meters).
        
    Returns:
        np.ndarray: Metric depth image (H x W).
    """
    metric_depth = depth_image.astype(np.float32) / scale  # Convert to meters
    return metric_depth

def generate_point_cloud(depth_image_path, color_image_path, camera_intrinsics, max_depth, depth_scale=1000.0):
    """
    Generate a point cloud from a depth image and a color image using Open3D.
    
    Args:
        depth_image_path (str): Path to the depth image file.
        color_image_path (str): Path to the color image file.
        camera_intrinsics (dict): Camera intrinsics with keys 'fx', 'fy', 'cx', 'cy'.
        max_depth (float): Maximum depth value in meters.
        depth_scale (float): Scale to convert depth image to metric depth.
        
    Returns:
        o3d.geometry.PointCloud: The generated point cloud.
    """
    # Load the depth image and color image
    color_image, raw_depth_image = preprocess_images(color_image_path, depth_image_path)

    # Convert raw depth image to a NumPy array
    depth_image_np = np.asarray(raw_depth_image)

    # Convert depth to metric depth
    metric_depth_image_np = depth_to_metric_depth(depth_image_np, scale=depth_scale)

    # Reconvert the metric depth image back to Open3D format
    metric_depth_image = o3d.geometry.Image(metric_depth_image_np)

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.asarray(color_image)),
        metric_depth_image,
        depth_trunc=max_depth,  # Truncate depth at max_depth
        convert_rgb_to_intensity=False
    )

    # Define the camera intrinsics
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
    width=392, height=512, fx=574.3343, fy=574.3343, cx=319.5, cy=239.5
    )

    # Generate the point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_matrix
    )

    # Flip the point cloud to align it with Open3D's coordinate system
    point_cloud.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

    return point_cloud

# Example usage
if __name__ == "__main__":
    # Paths to the depth and color images
    for i in range(28,100):
        depth_image_path = f"/home/ur10/Downloads/depth_map_new/depth_{i}.npy"  # Grayscale depth map
        color_image_path = f"/home/ur10/Downloads/depth_map_new/image_{i}.png"
        output_image_path = f"/home/ur10/Downloads/depth_map_new/depth_image_{i}.png"  # RGB image
        # npy_to_colorized_depth_image(depth_image_path, output_image_path)
        # Camera intrinsics (example values, replace with your actual data)
        camera_intrinsics = {
            'fx': 525.0,  # Focal length in x
            'fy': 525.0,  # Focal length in y
            'cx': 319.5,  # Principal point x
            'cy': 239.5   # Principal point y
        }

        # Maximum depth and depth scale
        max_depth = 5.0  # Maximum depth in meters
        depth_scale = 1000.0  # Depth scale: convert mm to meters

        # Generate the point cloud
        pcd = generate_point_cloud(depth_image_path, color_image_path, camera_intrinsics, max_depth, depth_scale)

        # Visualize the point cloud

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)

        # # Get the view control and set parameters
        # view_control = vis.get_view_control()
        # view_control.set_zoom(2)  # Adjust zoom level
        # view_control.set_front([0,0,-1])  # Set the front direction
        # view_control.set_lookat(lookat)  # Set the lookat point
        # view_control.set_up(up)  # Set the up vector

        # vis.run()
        # vis.destroy_window()
        o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd],
        {  # Callback for setting camera parameters
            ord("Z"): lambda vis: set_camera_zoom(vis, zoom=4.5),
        },
    )

        # Save the point cloud to a file
        # o3d.io.write_point_cloud("output_point_cloud.ply", pcd)
        print("Point cloud saved to 'output_point_cloud.ply'")
