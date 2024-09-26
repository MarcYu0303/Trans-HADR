import numpy as np
import torch
from scipy.ndimage import label, find_objects
import open3d as o3d

# from graspnet.utils import utils, sample
import numpy as np
# import open3d as o3d
import matplotlib.pyplot as plt

def keep_largest_component(pred_mask):
    # Function to keep the largest component for a single channel mask
    def largest_component(single_channel_mask):
        labeled_array, num_features = label(single_channel_mask)
        if num_features == 0:
            return single_channel_mask  # Return original mask if no features are found
        max_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
        return (labeled_array == max_label).astype(np.float32)
    
    # Convert PyTorch tensor to NumPy array if it's not already
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze(0).detach().cpu().numpy()
    
    # Apply the largest component function to each channel separately
    largest_masks = np.array([largest_component(channel) for channel in pred_mask])

    # Convert back to PyTorch tensor if needed
    return torch.from_numpy(largest_masks).unsqueeze(0)

def save_point_cloud(xyz, color, file_path):
    assert xyz.shape[0] == color.shape[0]
    assert xyz.shape[1] == color.shape[1] == 3
    ply_file = open(file_path, 'w')
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('element vertex {}\n'.format(xyz.shape[0]))
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('property uchar red\n')
    ply_file.write('property uchar green\n')
    ply_file.write('property uchar blue\n')
    ply_file.write('end_header\n')
    for i in range(xyz.shape[0]):
        ply_file.write('{:.3f} {:.3f} {:.3f} {} {} {}\n'.format(
                                xyz[i,0], xyz[i,1], xyz[i,2],
                                color[i,0], color[i,1], color[i,2]
                            )
        )
        
def visualize_ply(file_path):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    
def get_color_plasma(x):
    return tuple([float(1 - x), float(x), float(0)])
    
# def draw_scene_with_open3d(pc,
#                            grasps=[],
#                            grasp_scores=None,
#                            grasp_color=None,
#                            gripper_color=(0, 1, 0),
#                            mesh=None,
#                            show_gripper_mesh=False,
#                            grasps_selection=None,
#                            visualize_diverse_grasps=False,
#                            min_seperation_distance=0.03,
#                            pc_color=None,
#                            plasma_coloring=False,
#                            target_cps=None,
#                            show_hand_keypoints=False,
#                            hand_keypoints=None):
#     """
#     Draws the 3D scene using Open3D.
#     """
#     max_grasps = 100
#     grasps = np.array(grasps)

#     if grasp_scores is not None:
#         grasp_scores = np.array(grasp_scores)
    
#     if len(grasps) > max_grasps:
#         print('Downsampling grasps, there are too many')
#         chosen_ones = np.random.randint(low=0,
#                                         high=len(grasps),
#                                         size=max_grasps)
#         grasps = grasps[chosen_ones]
#         if grasp_scores is not None:
#             grasp_scores = grasp_scores[chosen_ones]

#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     ## Visualize mesh
#     if mesh is not None:
#         if isinstance(mesh, list):
#             for m in mesh:
#                 vis.add_geometry(m)
#         else:
#             vis.add_geometry(mesh)
    
#     ## Visualize point-cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc)
#     # Check if custom colors are to be applied
#     if pc_color is None:
#         if plasma_coloring:
#             # Use a color map, normalize Z values between 0 and 1 for colormap application
#             z_values = pc[:, 2]
#             z_norm = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
#             colors = plt.get_cmap('plasma')(z_norm)[:, :3]  # Ignoring alpha channel
#             pcd.colors = o3d.utility.Vector3dVector(colors)
#         else:
#             # Apply a static color
#             colors = np.tile([0.1, 0.1, 1], (pc.shape[0], 1))
#             pcd.colors = o3d.utility.Vector3dVector(colors)
#     else:
#         # Apply provided colors, assuming they are in an acceptable format
#         if plasma_coloring:
#             # Interpret the first channel of pc_color for plasma coloring
#             colors = plt.get_cmap('plasma')(pc_color[:, 0])[:, :3]
#         else:
#             # Use the provided colors directly, assuming they are normalized RGB
#             colors = pc_color / 255
#         pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     ## Visualize gripper poses
#     grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)
#     grasp_pc[2, 2] = 0.059
#     grasp_pc[3, 2] = 0.059

#     mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

#     modified_grasp_pc = []
#     modified_grasp_pc.append(np.zeros((3, ), np.float32))
#     modified_grasp_pc.append(mid_point)
#     modified_grasp_pc.append(grasp_pc[2])
#     modified_grasp_pc.append(grasp_pc[4])
#     modified_grasp_pc.append(grasp_pc[2])
#     modified_grasp_pc.append(grasp_pc[3])
#     modified_grasp_pc.append(grasp_pc[5])

#     grasp_pc = np.asarray(modified_grasp_pc)    

#     if grasp_scores is not None:
#         indexes = np.argsort(-np.asarray(grasp_scores))
#     else:
#         indexes = range(len(grasps))

#     print('draw scene, the number of grasps is:', len(grasps))

#     selected_grasps_so_far = []
#     removed = 0

#     if grasp_scores is not None:
#         min_score = np.min(grasp_scores)
#         max_score = np.max(grasp_scores)
#         top5 = np.array(grasp_scores).argsort()[-5:][::-1]

#     for ii in range(len(grasps)):
#         i = indexes[ii]
#         if grasps_selection is not None:
#             if grasps_selection[i] == False:
#                 continue

#         g = grasps[i]
#         is_diverse = True
#         for prevg in selected_grasps_so_far:
#             distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])

#             if distance < min_seperation_distance:
#                 is_diverse = False
#                 break

#         if visualize_diverse_grasps:
#             if not is_diverse:
#                 removed += 1
#                 continue
#             else:
#                 if grasp_scores is not None:
#                     print('selected', i, grasp_scores[i], min_score, max_score)
#                 else:
#                     print('selected', i)
#                 selected_grasps_so_far.append(g)

#         if isinstance(gripper_color, list):
#             pass
#         elif grasp_scores is not None:
#             normalized_score = (grasp_scores[i] -
#                                 min_score) / (max_score - min_score + 0.0001)
#             if grasp_color is not None:
#                 gripper_color = grasp_color[ii]
#             else:
#                 gripper_color = get_color_plasma(normalized_score)

#             if min_score == 1.0:
#                 gripper_color = (0.0, 1.0, 0.0)
    
#         if show_gripper_mesh:
#             gripper_mesh = o3d.io.read_triangle_mesh('./graspnet/gripper_models/panda_gripper.obj')
#             gripper_mesh.transform(g)
#             gripper_mesh.paint_uniform_color(gripper_color)
#             vis.add_geometry(gripper_mesh)
#         else:
#             pts = np.matmul(grasp_pc, g[:3, :3].T) + np.expand_dims(g[:3, 3], 0)
            
#             lines = [[i, i + 1] for i in range(pts.shape[0] - 1)]
#             line_set = o3d.geometry.LineSet()
#             line_set.points = o3d.utility.Vector3dVector(pts)
#             line_set.lines = o3d.utility.Vector2iVector(lines)
#             if isinstance(gripper_color, list):
#                 colors = [gripper_color[i] for _ in range(len(lines))]
#             else:
#                 colors = [gripper_color for _ in range(len(lines))]
#             line_set.colors = o3d.utility.Vector3dVector(colors)
#             vis.add_geometry(line_set)
            
#             if target_cps is not None:
#                 pass
#                 # TBD
    
    
#     print('removed {} similar grasps'.format(removed))
    
#     # add the origin
#     origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#     vis.add_geometry(origin)
    
#     # visualize the point-cloud
#     vis.add_geometry(pcd)
    
#     # visualize the hand keypoints
#     if show_hand_keypoints and hand_keypoints is not None:
#         print('here')
#         kpts_pcd, kpts_line_set = visualize_hand_keypoints(hand_keypoints)
#         vis.add_geometry(kpts_pcd)
#         vis.add_geometry(kpts_line_set)
    
#     vis.run()
#     vis.destroy_window()


def visualize_hand_keypoints(pred_keypoint_xyz):
    connections = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], 
                [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]
    connections_color = ['red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue',
                        'yellow', 'yellow', 'yellow', 'cyan', 'cyan', 'cyan', 'red', 'green', 'blue', 'yellow', 'cyan']
    points_color = ['purple', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green',
                    'blue', 'blue', 'blue', 'blue', 'yellow', 'yellow', 'yellow', 'yellow', 'cyan', 'cyan', 'cyan', 'cyan']   
    
    # Create an Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    
    # Assign keypoints to the PointCloud
    pcd.points = o3d.utility.Vector3dVector(pred_keypoint_xyz)
    
    # Create an array to store the colors of the points
    point_colors = np.array([{
        'purple': [0.5, 0, 0.5], 'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
        'yellow': [1, 1, 0], 'cyan': [0, 1, 1]
    }[color] for color in points_color])
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Create lines for connections
    lines = connections
    line_colors = np.array([{
        'purple': [0.5, 0, 0.5], 'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
        'yellow': [1, 1, 0], 'cyan': [0, 1, 1]
    }[color] for color in connections_color])
    
    # Create a LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pred_keypoint_xyz)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    
    return pcd, line_set