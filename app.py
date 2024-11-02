import numpy as np
import cv2 as cv
import gradio as gr

# Filter functions
def apply_filter(image, filter_type, brightness=0, blur_level=1):
    if filter_type == "Grayscale":
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif filter_type == "Blur":
        blur_value = max(1, 2 * int(blur_level) + 1)
        return cv.GaussianBlur(image, (blur_value, blur_value), 0)
    elif filter_type == "Edge Detection":
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return cv.Canny(gray_image, 100, 200)
    elif filter_type == "Brightness Increase":
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv.add(hsv[:, :, 2], brightness)
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    elif filter_type == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_image = cv.transform(image, sepia_filter)
        sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
        return sepia_image
    elif filter_type == "Invert Colors":
        return cv.bitwise_not(image)
    elif filter_type == "Sharpen":
        sharpen_filter = np.array([[0, -1, 0], 
                                   [-1, 5,-1], 
                                   [0, -1, 0]])
        return cv.filter2D(image, -1, sharpen_filter)
    else:
        return image  # Return original image by default

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("Upload an image and select a filter to apply.")
    gr.Markdown('Drop the file..')
    # Input and Output components
    image_input = gr.Image(type="numpy", label="Input Image")
    filter_buttons = gr.Radio(["Grayscale", "Blur", "Edge Detection", "Brightness Increase", "Sepia", "Invert Colors", "Sharpen"],
                              label="Choose a Filter", interactive=True)
    brightness_slider = gr.Slider(0, 100, step=1, label="Brightness Level", visible=False)
    blur_slider = gr.Slider(1, 20, step=1, label="Blur Level", visible=False)
    image_output = gr.Image(type="numpy", label="Output Image")
    
    # Update slider visibility based on filter type
    def update_visibility(filter_type):
        return {
            brightness_slider: gr.update(visible=(filter_type == "Brightness Increase")),
            blur_slider: gr.update(visible=(filter_type == "Blur")),
        }

    # Trigger live updates when sliders or filter type change
    filter_buttons.change(fn=update_visibility, inputs=filter_buttons, outputs=[brightness_slider, blur_slider])
    filter_buttons.change(fn=apply_filter, inputs=[image_input, filter_buttons, brightness_slider, blur_slider], outputs=image_output)
    brightness_slider.change(fn=apply_filter, inputs=[image_input, filter_buttons, brightness_slider, blur_slider], outputs=image_output)
    blur_slider.change(fn=apply_filter, inputs=[image_input, filter_buttons, brightness_slider, blur_slider], outputs=image_output)

if __name__ == "__main__":
    demo.launch()
