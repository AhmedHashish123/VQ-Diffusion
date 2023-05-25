import streamlit as st
import os
from inference_VQ_Diffusion import VQ_Diffusion
import shutil
from super_image import EdsrModel, ImageLoader
from PIL import Image

@st.cache_resource
def create_diffusion_model():
    VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_pretrained.pth')
    return VQ_Diffusion_model

@st.cache_resource
def create_sr_model():
    sr_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    return sr_model

def set_stage(stage):
    st.session_state.stage = stage

# Define the Streamlit app
def main():

    VQ_Diffusion_model = create_diffusion_model()
    super_resolution_model = create_sr_model()


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/friendly-robot-artist-studio-his-easel-painting-paints-while-working-white-background-neural-network-ai-generated-art_636705-8307.jpg?w=2000");
             background-attachment: fixed;
             background-size: 100% 100%
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

    st.title(":red[Text-to-Image Generation]")
    
    tabs_font_css = """
    <style>
    div[class*="stTextInput"] label p {
    font-size: 20px;
    color: blue;
    font-weight: bold;
    }

    div[class*="stSelectbox"] label p {
    font-size: 20px;
    color: blue;
    font-weight: bold;
    }
    </style>
    """

    st.write(tabs_font_css, unsafe_allow_html=True)

    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    input_text = st.text_input("Please enter a description for the image", "an elephant walking in muddy water")

    n_images = st.selectbox("How many images do you want? (The higher the number, the more time it takes for the AI to draw them.)",("1", "2", "3", "4", "5"))
    st.button('Generate Image', on_click=set_stage, args=(1,))

    if st.session_state.stage == 1:

        path = "RESULT/"+input_text+"/"
        if(os.path.isdir(path)): # Directory already exists. We should delete it
            shutil.rmtree(path)

        onscreen = st.empty()
        onscreen.header(':violet[Generating Image...] :hourglass:')

        VQ_Diffusion_model.inference_generate_sample_with_condition(input_text, truncation_rate=0.86, save_root="RESULT", batch_size=int(n_images))
        
        images = os.listdir(path)

        for image_path in images:
            image_file = open(path+'/'+image_path,'rb')
            image_bytes = image_file.read()
            st.image(image_bytes)

        onscreen.empty()
        onscreen.header(':fireworks: :green[Done!] :fireworks:')

        onscreen2 = st.empty()
        onscreen2.header(':violet[This AI cannot draw high resolutions images] :disappointed:"')

        onscreen3 = st.empty()
        onscreen3.header(':red[But another AI does...] :face_with_rolling_eyes:')

        st.button('Increase Resolution', on_click=set_stage, args=(2,))

    if st.session_state.stage == 2:

        onscreen = st.empty()
        onscreen.header(':violet[Increasing Resolution...] :hourglass:')

        path = "RESULT/"+input_text+"/"
        images = os.listdir(path)

        for image_path in images:
            image_file = Image.open(path+'/'+image_path)
            inputs = ImageLoader.load_image(image_file)
            preds = super_resolution_model(inputs)
            image_path_without_extension = os.path.splitext(path+'/'+image_path)[0]
            image_path_without_extension = image_path_without_extension + "_hr"
            ImageLoader.save_image(preds, image_path_without_extension + ".png")


        images = os.listdir(path)
        for image_path in images:
            if("_hr" in image_path):
                image_file = open(path+'/'+image_path,'rb')
                image_bytes = image_file.read()
                st.image(image_bytes)

        onscreen.empty()
        onscreen.header(':fireworks: :green[Done!] :fireworks:')


    st.button('Reset', on_click=set_stage, args=(0,))


# Run the app
if __name__ == "__main__":
    main()



