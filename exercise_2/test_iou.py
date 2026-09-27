def compute_iou(box1, box2):
    """
    Tính Intersection over Union (IoU) của hai bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        iou: float
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def test_iou():
    # Test case 1: Identical boxes
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert compute_iou(box1, box2) == 1.0, f"Expected 1.0, got {compute_iou(box1, box2)}"

    # Test case 2: No intersection
    box1 = [0, 0, 10, 10]
    box2 = [20, 20, 30, 30]
    assert compute_iou(box1, box2) == 0.0, f"Expected 0.0, got {compute_iou(box1, box2)}"

    # Test case 3: Partial intersection
    # Intersection: [20, 20, 30, 30] -> Area = 10*10 = 100
    # Box1: [10, 10, 30, 30] -> Area = 20*20 = 400
    # Box2: [20, 20, 40, 40] -> Area = 20*20 = 400
    # Union = 400 + 400 - 100 = 700
    # IoU = 100 / 700 = 1/7 ≈ 0.142857
    box1 = [10, 10, 30, 30]
    box2 = [20, 20, 40, 40]
    iou = compute_iou(box1, box2)
    assert abs(iou - 0.142857) < 1e-5, f"Expected ~0.142857, got {iou}"

    print("All tests passed!")

if __name__ == "__main__":
    test_iou()
