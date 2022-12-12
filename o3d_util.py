import colorsys
import open3d as o3d


def visualize_pcd(pcd):
    assert pcd.shape[1] == 3, 'Invalid point cloud shape!'

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pcd_o3d])


def create_o3d_pcd(pcd):
    assert pcd.shape[1] == 3, 'Invalid point cloud shape!'
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    return pcd_o3d

def visualize_multiple_pcd(pcd_list):

    pcd_o3d_list = []
    num_pcd = len(pcd_list)
    for i, pcd in enumerate(pcd_list):
        assert pcd.shape[1] == 3, 'Invalid point cloud shape!'

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        (r, g, b) = colorsys.hsv_to_rgb((1/(num_pcd)) * i, 1, 1)
        pcd_o3d.paint_uniform_color([r, g, b])
        pcd_o3d_list.append(pcd_o3d)
    o3d.visualization.draw_geometries(pcd_o3d_list)
