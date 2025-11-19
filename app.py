import streamlit as st
from io import BytesIO
from PIL import Image
import json
import re
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


import os
import pandas as pd
import datetime
import tempfile

########################################
# Accessing HF Token
########################################
# hf_token = userdata.get('HF_TOKEN')
hf_token = st.secrets["OPENAI_API_KEY"]
client = InferenceClient(api_key= hf_token)

# drive.mount('/content/drive')

# # old_logic
# def convert_pdf_to_markdown(file):
#   converter = DocumentConverter()
#   result = converter.convert(file)
#   md_content = result.document.export_to_markdown()
#   return md_content

def convert_pdf_to_markdown(uploaded_file):
    file_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    pdf_options = PdfPipelineOptions(
        do_ocr=False,              #no RapidOCR
        do_table_structure=True,   
        do_picture_classification=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )

    try:
        result = converter.convert(tmp_path)
        md_content = result.document.export_to_markdown()
    finally:
        os.unlink(tmp_path)

    return md_content



def main():
  st.set_page_config( layout = "wide" ,page_title="Flyer Content Extraction")
  st.title(""":red[Flyer Content Extraction]""")

  st.markdown("### Upload your flyers below:")
  uploaded_files = st.file_uploader("Choose one or more PDF flyers", type="pdf", accept_multiple_files=True)

  extract_button = st.button("Extract Content")

  # if "messages" not in st.session_state:
  #     st.session_state["messages"] = []

  if "response" not in st.session_state:
      st.session_state["response"] = None

  if extract_button:
      if not uploaded_files:
          st.warning("Please upload at least one PDF before extracting.")
      else:
          with st.spinner("Reading flyers and extracting data..."):
              try:
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
        
                  completion = client.chat.completions.create(model="Qwen/Qwen2.5-14B-Instruct", # "Qwen/Qwen2.5-72B-Instruct",
                                                            messages=messages,
                                                            # max_tokens=6000,
                                                            temperature=0.0
                                                            #,stream=True
                                                            )
                
                  full_response = completion.choices[0].message.content
        
        
                  full_response_1 = full_response[7:-3]
                  full_response_1 = json.loads(full_response_1)
                  temp_data = pd.DataFrame(full_response_1)
        
                  st.session_state["response"] = temp_data



              except Exception as e:
                  st.error(f"Failed to extract content from flyer: {e}")



  if st.session_state["response"] is not None:
      temp_data = st.session_state["response"].copy()

      st.markdown("### Sample 5 rows of Extracted Data")
      st.write(temp_data.head(5))

      today = datetime.date.today().strftime("%Y-%m-%d")
      st.download_button("⬇️ Download full extracted data as CSV",data=temp_data.to_csv(index=False),
                         file_name=f"flyer_data_{today}.csv",mime="text/csv", )

      # ========= DATA PREP =========
      temp_data = temp_data.copy()
      expected_cols = ["Brand", "Quantity", "Price"]
      for col in expected_cols:
          if col not in temp_data.columns:
              st.warning(f"Column '{col}' not found in extracted data.")
              st.stop()

      temp_data["Price_clean"] = (temp_data["Price"].astype(str)
                                  .str.replace(r"[^\d.,]", "", regex=True)
                                  .str.replace(",", "", regex=False)
                                  .replace("", "0")
                                  .astype(float) )

      # ========= SIDEBAR FILTERS =========
      st.sidebar.title("🔍 Filters")

      all_brands = sorted(temp_data["Brand"].dropna().unique().tolist())
      selected_brands = st.sidebar.multiselect("Filter by brand:",
                                               options=all_brands,
                                               default=all_brands,)

      min_price = float(temp_data["Price_clean"].min())
      max_price = float(temp_data["Price_clean"].max())

      price_range = st.sidebar.slider("Price range (£):",
                                      min_value=round(min_price, 2),
                                      max_value=round(max_price, 2),
                                      value=(round(min_price, 2), round(max_price, 2)), )


      filtered_data = temp_data[temp_data["Brand"].isin(selected_brands) & temp_data["Price_clean"].between(price_range[0], price_range[1]) ]

      st.markdown("---")

      st.subheader("📄 Filtered Items")
      st.caption(f"Showing {len(filtered_data)} item(s) "
                 f"for {len(selected_brands)} brand(s) within £{price_range[0]:.2f} – £{price_range[1]:.2f}"      )
      st.dataframe(filtered_data, use_container_width=True)

      # Download filtered data
      st.download_button(
            "⬇️ Download filtered data as CSV",
            data=filtered_data.to_csv(index=False),
            file_name=f"flyer_data_filtered_{today}.csv",
            mime="text/csv",
        )


      # ========= QUICK INSIGHTS =========
      st.markdown("---")
      st.subheader("📊 Quick Insights")
      if not filtered_data.empty:
          # Top brand (within filters)
          brand_counts = (filtered_data["Brand"].value_counts().reset_index()  )

          brand_counts.columns = ["Brand", "Count"]

          if not brand_counts.empty:
              top_brand = brand_counts.iloc[0]
              col_top, col_total = st.columns(2)
              with col_top:
                  st.markdown("#### 🏆 Most frequent brand (filtered)")
                  st.metric(label="Brand", value=top_brand["Brand"],
                            delta=f"{int(top_brand['Count'])} items", )
        
              with col_total:
                  st.markdown("#### 📦 Total distinct brands (filtered)")
                  st.metric(label="Unique brands",value=int(filtered_data["Brand"].nunique()), )
               
              with st.expander("See all brands by frequency"):
                  st.dataframe(brand_counts, use_container_width=True)
          else:
              st.info("No brands found in the filtered data.")
      else:
          st.info("No data available under the current filters to compute brand stats.")
        
      # ========= SIZES / QUANTITIES BY BRAND =========
      st.markdown("#### 📦 Sizes by brand")
    
      if not filtered_data.empty:
          unique_brands_filtered = sorted(filtered_data["Brand"].dropna().unique().tolist())

          selected_brand_for_sizes = st.selectbox("Choose a brand to see all available sizes and their counts:",
                                                  unique_brands_filtered, )
        
          brand_subset = filtered_data[filtered_data["Brand"] == selected_brand_for_sizes]
        
          size_summary = ( brand_subset.groupby("Quantity", dropna=False).agg(
                    Count=("Quantity", "size"),
                    Min_price=("Price_clean", "min"),
                    Max_price=("Price_clean", "max"),
                ).reset_index().sort_values("Count", ascending=False))
        
          size_summary["Min_price"] = size_summary["Min_price"].map(lambda x: f"£{x:.2f}")
          size_summary["Max_price"] = size_summary["Max_price"].map(lambda x: f"£{x:.2f}")
        
          st.write(f"All sizes for **{selected_brand_for_sizes}** (filtered data):")
          st.dataframe(size_summary, use_container_width=True)
      else:
          st.info("No data to display sizes. Try relaxing your filters.")
        
      # ========= MOST / LEAST EXPENSIVE ITEMS =========
      st.markdown("#### 💰 Price extremes (within filters)")
        
      if not filtered_data.empty:
          valid_prices = filtered_data[filtered_data["Price_clean"].notna()]
          if not valid_prices.empty:
              max_row = valid_prices.loc[valid_prices["Price_clean"].idxmax()]
              min_row = valid_prices.loc[valid_prices["Price_clean"].idxmin()]

              col1, col2 = st.columns(2)
              
              with col1:
                  st.markdown("**Most expensive item**")
                  st.write(f"Brand: {max_row['Brand']}")
                  st.write(f"Quantity: {max_row['Quantity']}")
                  st.write(f"Price: £{max_row['Price_clean']:.2f}")
        
              with col2:
                  st.markdown("**Least expensive item**")
                  st.write(f"Brand: {min_row['Brand']}")
                  st.write(f"Quantity: {min_row['Quantity']}")
                  st.write(f"Price: £{min_row['Price_clean']:.2f}")
          else:
              st.info("No valid prices found to compute min/max.")
      else:
          st.info("No data available to compute price extremes under current filters.")
        
      # ========= SIMPLE CHARTS =========
      st.markdown("---")
      st.subheader("📈 Visualizations")
        
      if not filtered_data.empty:
          # Brand frequency bar chart
          chart_brand_counts = ( filtered_data["Brand"].value_counts().reset_index()
                                .rename(columns={"index": "Brand", "Brand": "Count"})
                                .sort_values("Count", ascending=False)       )
        
          st.markdown("##### Brand frequency (filtered)")
          st.bar_chart(chart_brand_counts.set_index("Brand")["Count"],
                use_container_width=True,    )
        
          # Price distribution
          st.markdown("##### Price distribution (filtered)")
          st.line_chart( filtered_data["Price_clean"].sort_values().reset_index(drop=True),
                use_container_width=True,      )
      else:
          st.info("No data to display charts. Try adjusting the filters.")

  else:
      st.info("Upload flyers and click **Extract Content** to see results.")
                

if __name__ == "__main__":
  main()
