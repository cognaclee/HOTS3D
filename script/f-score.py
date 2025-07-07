import os
import numpy as np
import open3d as o3d
import time
def normalize_mesh(mesh):
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    
    bbox = new_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    new_mesh.translate(-center)
    
    max_extent = np.max(extent)
    if max_extent > 1e-6:
        new_mesh.scale(1.0 / max_extent, center=np.zeros(3))
    
    return new_mesh

def sample_mesh_points(mesh, num_points=100000):
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=7)
    return np.asarray(pcd.points)

def calculate_f_score(mesh_pred, mesh_gt, num_points=100000):
    """计算F-Score"""
    norm_pred = normalize_mesh(mesh_pred)
    norm_gt = normalize_mesh(mesh_gt)
    
    bbox_gt = norm_gt.get_axis_aligned_bounding_box()
    tau = np.linalg.norm(bbox_gt.get_extent()) * 0.08
    
    pred_points = sample_mesh_points(norm_pred, num_points)
    gt_points = sample_mesh_points(norm_gt, num_points)
    
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
    pred_tree = o3d.geometry.KDTreeFlann(pred_pcd)
    
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
    
    precision_dist = [np.linalg.norm(p - gt_points[gt_tree.search_knn_vector_3d(p, 1)[1][0]]) for p in pred_points]
    recall_dist = [np.linalg.norm(p - pred_points[pred_tree.search_knn_vector_3d(p, 1)[1][0]]) for p in gt_points]
    
    precision = np.mean(np.array(precision_dist) < tau)
    recall = np.mean(np.array(recall_dist) < tau)
    f_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f_score

def get_sorted_files(path, is_pred=False):
    files = [f for f in os.listdir(path) if f.endswith(".obj")]
    if is_pred:
        files.sort(key=lambda x: (int(x.split("_")[0]), x))
    else:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(path, f) for f in files]

def main():
    start_time = time.time()
    pred_dir = "./results/text2shape/mesh/"
    gt_dir = "./datasets/text2shape/mesh/"
    
    pred_files = get_sorted_files(pred_dir, is_pred=True)
    gt_files = get_sorted_files(gt_dir, is_pred=False)
    
    if len(pred_files) != len(gt_files):
        print("Unmatched number of files")
        return
    
    total_precision, total_recall, total_f_score = 0, 0, 0
    num_files = len(pred_files)
    
    for pred_path, gt_path in zip(pred_files, gt_files):
        mesh_pred = o3d.io.read_triangle_mesh(pred_path)
        mesh_gt = o3d.io.read_triangle_mesh(gt_path)
        
        if not mesh_pred.has_vertices() or not mesh_gt.has_vertices():
            print(f"skip invalid mesh file: {pred_path} or {gt_path}")
            continue
        
        precision, recall, f_score = calculate_f_score(mesh_pred, mesh_gt)
        total_precision += precision
        total_recall += recall
        total_f_score += f_score
    
    avg_precision = total_precision / num_files
    avg_recall = total_recall / num_files
    avg_f_score = total_f_score / num_files
    
    print(f"Average Precision：{avg_precision:.4f}")
    print(f"Average Recall：{avg_recall:.4f}")
    print(f"Average F-Score：{avg_f_score:.4f}")
    end_time = time.time()
    time = end_time - start_time
    print(f"Run time is {time} !!")
if __name__ == "__main__":
    main()
