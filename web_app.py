import streamlit as st
import os
from inference_VQ_Diffusion import VQ_Diffusion
import shutil

@st.cache_resource
def create_diffusion_model():
    VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_pretrained.pth')
    return VQ_Diffusion_model


VQ_Diffusion_model = create_diffusion_model()

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


input_text = st.text_input(":red[**Please enter a description for the image**]", "an elephant walking in muddy water")

if st.button('Generate Image'):

    onscreen = st.empty()
    onscreen.header(':violet[Generating Image...] :hourglass:')

    path = "RESULT/"+input_text+"/"
    if(os.path.isdir(path)): # Directory already exists. We should delete it
        shutil.rmtree(path)

    VQ_Diffusion_model.inference_generate_sample_with_condition(input_text, truncation_rate=0.86, save_root="RESULT", batch_size=1)
    
    images = os.listdir(path)

    for image_path in images:
        image_file = open(path+'/'+image_path,'rb')
        image_bytes = image_file.read()
        st.image(image_bytes)

    
    
    onscreen.empty()
    onscreen.header(':fireworks: :green[Done!] :fireworks:')
