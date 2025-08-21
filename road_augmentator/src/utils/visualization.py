def visualize_zones(image, zones):
    for zone in zones:
        x1, y1, x2, y2 = zone["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Placement Zones", image)
    cv2.waitKey(0)
    
    