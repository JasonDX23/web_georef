import streamlit as st
import os
import cv2
import numpy as np
import geopandas as gpd
import re
import rasterio
from rasterio.transform import from_bounds
from io import BytesIO
from streamlit_folium import folium_static
import folium
import zipfile
gdf = gpd.read_file("soi_osm_sheet_index.geojson.json" )

def process_single_image(image, filename):
    try:
        # Step 1: Extract black pixels (black_mask)
        tolerance = 150
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([tolerance, tolerance, tolerance])
        black_mask = cv2.inRange(image, lower_bound, upper_bound)

            # Step 2: Detect long lines (HoughLinesP) on the black mask
        min_line_length_ratio = 0.25
        threshold = 50
        max_line_gap = 10
        height, width = black_mask.shape
        min_line_length = int(width * min_line_length_ratio)
        lines = cv2.HoughLinesP(black_mask, rho=1, theta=np.pi / 180, threshold=threshold, 
                                    minLineLength=min_line_length, maxLineGap=max_line_gap)

            # Step 3: Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        angular_tolerance = 5
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            if abs(angle) <= angular_tolerance or abs(angle - 180) <= angular_tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(angle - 90) <= angular_tolerance:
                vertical_lines.append((x1, y1, x2, y2))

            # Step 4: Find outer frame corners from the horizontal and vertical lines
        horizontal_lines.sort(key=lambda line: (line[1] + line[3]) // 2)
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        vertical_lines.sort(key=lambda line: (line[0] + line[2]) // 2)
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]

        corners = [(left_line[0], top_line[1]), (right_line[2], top_line[1]),
                (left_line[0], bottom_line[3]), (right_line[2], bottom_line[3])]

            # Step 5: Refine the corners using Sobel gradients
        sobel_x = cv2.Sobel(black_mask, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(black_mask, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        refined_corners = []
        roi_size = 15
        for corner in corners:
            x, y = corner
            x, y = int(x), int(y)
            x_start, x_end = max(0, x - roi_size // 2), min(black_mask.shape[1], x + roi_size // 2)
            y_start, y_end = max(0, y - roi_size // 2), min(black_mask.shape[0], y + roi_size // 2)
            roi = gradient_magnitude[y_start:y_end, x_start:x_end]
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
            refined_x = x_start + max_loc[0]
            refined_y = y_start + max_loc[1]
            refined_corners.append((refined_x, refined_y))

            # Step 6: Estimate inner corners based on outer corners
            frame_width_mm = 12
            pixel_distance = abs((top_line[1] + top_line[3]) // 2 - (bottom_line[1] + bottom_line[3]) // 2)
            resolution = pixel_distance / 391
            offset = frame_width_mm * resolution
            inner_corners = [(x + offset, y + offset) if i in [0] else
                            (x - offset, y + offset) if i in [1] else
                            (x + offset, y - offset) if i in [2] else
                            (x - offset, y - offset)
                            for i, (x, y) in enumerate(corners)]

            # Refine inner corners
            refined_inner_corners = []
            for corner in inner_corners:
                x, y = corner
                x, y = int(x), int(y)
                x_start, x_end = max(0, x - roi_size // 2), min(black_mask.shape[1], x + roi_size // 2)
                y_start, y_end = max(0, y - roi_size // 2), min(black_mask.shape[0], y + roi_size // 2)
                roi = gradient_magnitude[y_start:y_end, x_start:x_end]
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
                refined_x = x_start + max_loc[0]
                refined_y = y_start + max_loc[1]
                refined_inner_corners.append((refined_x, refined_y))

            # Step 7: Create a GeoTIFF output
            original_image = image
            corners_array = np.array(refined_inner_corners)

            x_min, y_min = corners_array.min(axis=0)
            x_max, y_max = corners_array.max(axis=0)

            # Step 8: Format and check if the image exists in the GeoJSON file
            pattern = r"(\d{2} [A-Z])\](\d{2})"
            match = re.search(pattern, filename)
            if match:
                main_part = match.group(1).replace(' ', '')
                number_part = str(int(match.group(2)))
                formatted = f"{main_part}/{number_part}"

            cropped_image = original_image[int(y_min):int(y_max), int(x_min):int(x_max)]
            if formatted in gdf['EVEREST_SH'].values:
                geometry = gdf.loc[gdf['EVEREST_SH'] == formatted, 'geometry'].values[0]
                print("Geometry found:", geometry)
                height, width = cropped_image.shape[:2]
                minx, miny, maxx, maxy = geometry.bounds
                transform = from_bounds(minx, miny, maxx, maxy, width, height)
                crs = "epsg:4326"  # modify if needed

                # Create a geotiff file using rasterio
                tiff_buffer = BytesIO()
                with rasterio.open(tiff_buffer, 'w', 
                    driver='GTiff', height=cropped_image.shape[0], width=cropped_image.shape[1],
                    count=3, dtype='uint8', crs=crs, transform=transform) as dst:
                    # If the cropped_image is in RGB format, you may want to write R, G, B in the correct order.
                    # Check if the image is in BGR (common in OpenCV) and adjust accordingly.
                    
                    # If BGR format (common for OpenCV), convert to RGB:
                    if cropped_image.shape[2] == 3:
                        cropped_image = cropped_image[..., ::-1]  # Flip channels from BGR to RGB

                    # Write the individual channels in the correct band order (R, G, B).
                    for i in range(1, 4):  # Assuming 3 bands (RGB)
                        dst.write(cropped_image.transpose(2, 0, 1)[i-1], i)
                tiff_buffer.seek(0)
                return tiff_buffer
    except Exception as e:
        st.error(f"error processing {filename}: {e}")
        return None

def create_zip_of_processed_images(processed_files):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, tiff_buffer in processed_files.items():
            zf.writestr(f"{filename}.tif", tiff_buffer.getvalue())
    memory_file.seek(0)
    return memory_file

def main():
    st.title("Automatic Georeferencer")
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Images"):
            processed_files = {}
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    filename = uploaded_file.name
                    
                    # Convert uploaded file to image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Process the image
                    tiff_data = process_single_image(image, filename)
                    
                    if tiff_data:
                        processed_files[os.path.splitext(filename)[0]] = tiff_data
                    else:
                        st.error(f"Failed to process {filename}")
                        
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if processed_files:
                # Create zip file of processed images
                zip_file = create_zip_of_processed_images(processed_files)
                
                # Create download button
                st.download_button(
                    label="Download All Processed GeoTIFFs",
                    data=zip_file,
                    file_name="processed_geotiffs.zip",
                    mime="application/zip"
                )
                
                st.success(f"Successfully processed {len(processed_files)} out of {len(uploaded_files)} images")

                st.markdown(
                        """
                        <style>
                        .signature {
                            position: fixed;
                            bottom: 20px;  /* Distance from bottom */
                            right: 20px;   /* Distance from right */
                            color: #888;   /* Gray color */
                            font-size: 14px;
                            font-style: italic;
                            z-index: 1000;
                        }
                        </style>
                        <div class="signature">Made by Jason Dsouza</div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()