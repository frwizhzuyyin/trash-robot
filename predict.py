import cv2
from ultralytics import YOLO

# Load your model (use the raw string r'' for path on Windows)
# The extra quotes have been removed: r"C:\..."
model = YOLO(r"C:\trash_robot\best.pt")

# Perform prediction and store the results, but DO NOT show the image automatically
results = model.predict(
    source=r"C:\trash_robot\test_image.jpg", # The extra quotes have been removed: r"C:\..."
    conf=0.05,
    show=False,
    save=False # Changed to False, as you are displaying manually and likely don't need a separate saved file
)

# --- MANUAL DISPLAY CODE STARTS HERE ---

if results:
    # 1. Get the annotated frame (the result with boxes/labels drawn)
    # results[0] contains the data for the first (and only) image processed
    annotated_frame = results[0].plot()

    # 2. Display the frame manually
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # 3. SETS THE WINDOW TO WAIT INDEFINITELY UNTIL A KEY IS PRESSED
    # 0 means the window will stay open until you press a key or close it manually.
    key = cv2.waitKey(0) # <--- CRITICAL CHANGE: Set to 0 

    # 4. Clean up
    cv2.destroyAllWindows()
else:
    print("Inference completed, but no results were returned.")