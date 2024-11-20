import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment
    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements
    check_requirements("lapx>=0.5.2")  # update to lap package from https://github.com/rathaROG/lapx
    import lap


class Matching:
    @staticmethod
    def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
        """
        Perform linear assignment using either the scipy or lap.lapjv method.
        """
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        if use_lap:
            # Use lap.lapjv
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        else:
            # Use scipy.optimize.linear_sum_assignment
            x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
            matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
            if len(matches) == 0:
                unmatched_a = list(np.arange(cost_matrix.shape[0]))
                unmatched_b = list(np.arange(cost_matrix.shape[1]))
            else:
                unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
                unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

        return matches, unmatched_a, unmatched_b

    @staticmethod
    def iou_distance(atracks: list, btracks: list) -> np.ndarray:
        """
        Compute cost based on Intersection over Union (IoU) between tracks.
        """
        if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
            btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
        if len(atlbrs) and len(btlbrs):
            if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
                ious = batch_probiou(
                    np.ascontiguousarray(atlbrs, dtype=np.float32),
                    np.ascontiguousarray(btlbrs, dtype=np.float32),
                ).numpy()
            else:
                ious = bbox_ioa(
                    np.ascontiguousarray(atlbrs, dtype=np.float32),
                    np.ascontiguousarray(btlbrs, dtype=np.float32),
                    iou=True,
                )
        return 1 - ious  # cost matrix

    @staticmethod
    def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
        """
        Compute distance between tracks and detections based on embeddings.
        """
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if cost_matrix.size == 0:
            return cost_matrix
        det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
        track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
        cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
        return cost_matrix

    @staticmethod
    def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
        """
        Fuses cost matrix with detection scores to produce a single similarity matrix.
        """
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        fuse_sim = iou_sim * det_scores
        return 1 - fuse_sim  # fuse_cost
