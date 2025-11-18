import streamlit as st
from io import BytesIO
from PIL import Image
import json
import re
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter
import os
import pandas as pd
import datetime


########################################
# Accessing HF Token
########################################
# hf_token = userdata.get('HF_TOKEN')
hf_token = ""
client = InferenceClient(api_key= hf_token)

# drive.mount('/content/drive')


def convert_pdf_to_markdown(file):
  converter = DocumentConverter()
  result = converter.convert(file)
  md_content = result.document.export_to_markdown()
  return md_content



def main():
  st.set_page_config( layout = "wide" ,page_title="Text Extraction")
  st.title(""":green[Text Extraction]""")

  st.write("<h3>Upload your flyers below:")
  uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

  extract_button = st.button("Extract Content")

  if "messages" not in st.session_state:
    st.session_state["messages"] = []

  if extract_button:
    try:
      if uploaded_files:
        st.session_state["messages"] = [{"role":"assistant", "content": "Reading the uploaded flyer/flyers"}]


        for file in uploaded_files:
          filename = '_'.join(file.name.split('.')[0].split()) + '_data'
          if filename not in st.session_state:
            st.session_state[filename] = None

        for file in uploaded_files:
          filename = '_'.join(file.name.split('.')[0].split()) + '_data'
          try:
            st.session_state[filename] = convert_pdf_to_markdown(file)
          except Exception as e:
            st.error(f"Failed to read file {filename} from flyer: {e}")

        input_flyer_content = []
        for key in st.session_state:
          if key.endswith('_data'):
            input_flyer_content.append(' ' + key + ' file contents are: ' +   st.session_state[key])

        print("Combined input flyer_contents are:", input_flyer_content)

        user_prompt = """ The provided flyer contains beverages information along with quantity and their pricing.
         The pattern is: first the brand name is provided like CARLING or sometimes two brands are provided together like BUDWEISER/STELLA.
         When two two brands are present, we need to split them into two different records.
         Next comes the quantity (likely starts with numbers like 6X4X440ML.
          In some scenarios two quantities are mentioned like 6X4X440ML/568ML.
          In this scenario where two quantities are mentioned, you will see two prices in next line.
          Based on number of quantities, prices are segmented by '/'. Prices are in pounds currency.
          For example, if there is a scenario like STELLA 6X4X440ML/568ML PMP and line below has prices like £19.99/£23.99,
           the final output should be in two rows.
           *first row with three columns- Brand: STELLA, Quantity:6X4X440ML, Price:£19.99.
           *second row with three columns- Brand: STELLA, Quantity:568ML, Price:£23.99.


          Note:
          1. PMP stands for Price-Marked Pack. It's not quantity. Quantity is generally numeric or alphanumeric.
           Extract records which has quantity only.
           2. Final output should contain three columns only. Brand, Quantity, Price.
           3. If there is no price for any item, DO NOT extract those records. Be wise.
           4. Make sure you read the entire markdown file.
            Read until the last line and extract all the info based on above instructions.
           5. Nothing more. Please output JSON format only. No explanations, no hallucinations. Just Json output only. """


        messages = [
        {
        "role": "system",
        "content": f"You are a specialized flyer specialist. your role is to extract information from any flyer that is provided to you in a markdown format. Here is the flyer data in markdown format: {input_flyer_content}"

         },
          {
          "role": "user",
          "content": f"please follow the user prompts below {user_prompt}" }]

        completion = client.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct",
                                                    messages=messages,
                                                    # max_tokens=6000,
                                                    temperature=0.0
                                                    #,stream=True
                                                    )
        
        full_response = completion.choices[0].message.content



        # content_parts = []
        # for chunk in completion:
        #   if chunk.choices:
        #     content = chunk.choices[0].delta.content or ""
        #     # print(content, end="", flush=True)
        #     content_parts.append(content)
        #   elif chunk.usage:
        #     print("\n--- Request Usage ---")
        #     print(f"Input Tokens: {chunk.usage.prompt_tokens}")
        #     print(f"Output Tokens: {chunk.usage.completion_tokens}")
        #     print(f"Total Tokens: {chunk.usage.total_tokens}")

        # full_response = "".join(content_parts)

        full_response_1 = full_response[7:-3]
        full_response_1 = json.loads(full_response_1)
        temp_data = pd.DataFrame(full_response_1)

        st.session_state["response"] = temp_data

        if "response" in st.session_state:
          today = datetime.date.today().strftime("%Y-%m-%d")
          col1,col2 = st.columns([4,1])
          with col1:
            download_button = st.download_button("Download CSV", data = temp_data.to_csv(index=False), file_name=f"flyer_data_{today}.csv")

        st.write("### Sample 5 rows of Extracted Data")
        st.write(temp_data.head(5))




    except Exception as e:
      st.error(f"Failed to extract content from flyer: {e}")



if __name__ == "__main__":
  main()
