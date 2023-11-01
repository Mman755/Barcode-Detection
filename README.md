A barcode detection system I implemented using Python. Given an input image it will attempt to detect the barcode in it using various algorithms from image processing such as Gaussian filtering, edge detection, dilation techniques etc.

# 'Barcode_detection.py' 
This barcode detection file has algorithms I manually implemented from scratch using plain python. It works well for images with a single barcode, however, it has limitations. It cannot detect more than one barcode in an image, even though the image might have multiple barcodes, and, it is quite slow. 

## Images
![Barcode2_output 7 27 14 PM](https://github.com/Mman755/Barcode-Detection/assets/100733144/833c26f8-c407-4ad2-8a3f-551dc8bfe9f9)
![Barcode3_output 7 27 14 PM](https://github.com/Mman755/Barcode-Detection/assets/100733144/4e346e6d-510b-4240-9182-b7e5dd28cde4)

# 'Extension.py'
This barcode detection file leverages OpenCV to detect barcodes. Essentially the process is the same as 'Barcode_detection.py' however the pipeline instead just uses OpenCV now, which is much faster and effecient. I also implemented mutliple barcode scanning for this pipeline.
![Multiple_barcodes_output 7 27 15 PM](https://github.com/Mman755/Barcode-Detection/assets/100733144/0e15ccad-e720-49e4-99cd-4e248a1616f5)

# Comparisons between the two pipelines
## 'Barcode_detection.py'
![CleanShot 2023-11-01 at 15 17 27](https://github.com/Mman755/Barcode-Detection/assets/100733144/bff508dd-cb10-468a-a796-58c992af1022)
## 'Extension.py'
![CleanShot 2023-11-01 at 15 17 46](https://github.com/Mman755/Barcode-Detection/assets/100733144/4eb62cd5-ae67-4b5a-99df-0ef6e25e5777)
## 'Speed comparison'
![CleanShot 2023-11-01 at 15 18 09](https://github.com/Mman755/Barcode-Detection/assets/100733144/a51177db-8a98-4fc1-877e-fb2c3da0dd94)



