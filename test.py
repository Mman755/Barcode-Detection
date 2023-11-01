import cv2
import pyzbar

def scan_barcodes():
    # Create a VideoCapture object to capture video from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the current frame from the camera
        _, frame = cap.read()

        # Convert the frame to grayscale for barcode detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and decode barcodes in the grayscale image
        barcodes = pyzbar.decode(gray)

        # Process each detected barcode
        for barcode in barcodes:
            # Extract the barcode data and type
            data = barcode.data.decode("utf-8")
            barcode_type = barcode.type

            # Print the barcode data and type to the console
            print("Barcode Data:", data)
            print("Barcode Type:", barcode_type)

            # Draw a rectangle around the barcode on the frame
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw barcode data and type labels on the frame
            text = f"{barcode_type}: {data}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Barcode Scanner", frame)

        # Wait for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the barcode scanning function
scan_barcodes()