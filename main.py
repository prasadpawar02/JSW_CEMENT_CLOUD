import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import BedrockClaude
from langchain.llms.bedrock import Bedrock 
from pandasai.responses.streamlit_response import StreamlitResponse
import os
import boto3
from botocore.config import Config
import glob
import openai
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key=os.getenv('Open_AI_key')

st.set_page_config(
    page_title="JSW_CEMENT CHATBOT",
    layout="wide",
)
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


# Configure AWS and Bedrock settings
st.title("JSW_CEMENT CHATBOT")
config = Config(read_timeout=1000)
client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', config=config)
llm = BedrockClaude(client)

# Initialize the new LLM model for summarization
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', config=config)

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0",
                  client=bedrock,
                  model_kwargs={'max_gen_len': 512,
                                'temperature': 0.5,
                                'top_p': 0.9})
    return llm

def initialize_smart_df(file):
    df = pd.read_csv(file)
    smart_df = SmartDataframe(df,
                          config={"llm": llm, "response_parser": StreamlitResponse, "save_charts": True,
                                  "save_charts_path": os.path.join(os.getcwd(), "plots")}
                          )

    return smart_df

# Provide a default CSV file path or use a pre-uploaded file path
default_file_path = r"C:\Users\prasad.pawar\OneDrive - Tridiagonal Solutions\Desktop\JWT\Final_JSW.csv"

smart_df = initialize_smart_df(default_file_path)

def get_answer(smart_df, user_query):
    df_prompt = f"""
  Given the user query: "{user_query}", follow these steps to retrieve the necessary information:
    1. **Filtering the Data:**
       - You must follow the {user_query} while performing the task.
       - Filter the dataframe to include only rows where ['JSW Brand' is 'Yes' or 'Benchmark Brand' is 'Yes'] and 'Evidential' is 'Yes'.
       - Convert all dates to a consistent format if needed (e.g., YYYY-MM-DD) before filtering.
       - Ensure the 'State' filter matches exactly with the state mentioned in the query (case-sensitive).
       - Filter the data to include only rows where the 'Date' is exactly one day before the date mentioned in the query. For example, if the query mentions April 20th, include only data of April 19th.
       - If no data is retrieved after filtering, log the intermediate dataframe to debug.

    2. **Assigning WSP Values:**
       - Create a new column 'JSW Brand WSP' and set its value to the WSP if 'JSW Brand' is 'Yes', otherwise set it to 0.
       - Create a new column 'Benchmark Brand WSP' and set its value to the WSP if 'Benchmark Brand' is 'Yes', otherwise set it to 0.
    
    3. **Calculating the WSP Difference:**
       - For each unique combination of district and date:
           - Identify the rows corresponding to 'JSW Brand' and 'Benchmark Brand'.
           - Calculate the WSP difference as 'JSW Brand WSP' - 'Benchmark Brand WSP'. If either 'JSW Brand WSP' or 'Benchmark Brand WSP' is 0, do not perform the comparison.
           - Fill this difference in the 'WSP Difference' column for both rows (JSW Brand and Benchmark Brand) for the same district and date.
           - After this step, the final data should have all three WSP values filled for that district and date combination.
           
    4. **Providing the Result:**
       - Output the result in a dataframe format with the following columns:
         - State
         - District
         - Date
         - Brand
         - NOP (Net Operating Profit)
         - JSW Brand WSP
         - Benchmark Brand WSP
         - WSP Difference (the difference between 'JSW Brand WSP' and 'Benchmark Brand WSP')
         - Evidential

    Ensure to follow the {user_query} to generate the desired output.
"""
    response = smart_df.chat(df_prompt)
    return response

def top3(smart_df, user_query):
    df_top3_pr = f"""
    - **Step 1: Filter the Data**
      - Filter the data to include only rows where either 'JSW Brand' is 'Yes' or 'Benchmark Brand' is 'Yes'.

    - **Step 2: Assigning WSP Values**
      - Create a new DataFrame where:
        - 'JSW Brand WSP' is set to the WSP value if 'JSW Brand' is 'Yes', otherwise it should be NaN.
        - 'Benchmark Brand WSP' is set to the WSP value if 'Benchmark Brand' is 'Yes', otherwise it should be NaN.
      - Group the data by 'State', 'District', and 'Date', and use `max()` to align the 'JSW Brand WSP' and 'Benchmark Brand WSP' in the same row.

    - **Step 3: Calculating the WSP Difference**
      - Create a new column 'Price Difference' as 'Benchmark Brand WSP' - 'JSW Brand WSP'.
      - Filter the rows to retain only those where both 'JSW Brand WSP' and 'Benchmark Brand WSP' are non-zero and 'Price Difference' is positive.

    - **Step 4: Identify Top Price Increase Opportunities**
      - Sort the data by 'Price Difference' in descending order.
      - For each state, select the top 3 districts with the largest price differences.

    - **Step 5: Prepare and Display the Final Table**
      - The final table should display the top 3 districts state-wise with the highest potential for JSW to increase its price.
      - Include the following columns:
        - State
        - District
        - JSW Brand WSP
        - Benchmark Brand WSP
        - Price Difference (Benchmark Brand WSP - JSW Brand WSP)
        - RSP (Retail Selling Price)
     """
    response = smart_df.chat(df_top3_pr)
    return response

def generate_l(smart_df, user_query):
    prompt = """
    - **Step 1: Filter the Data**
      - Filter the data to include only rows where 'JSW Brand' is 'Yes' or 'Benchmark Brand' is 'Yes'.

    - **Step 2: Count Evidential Pricing Uploads**
      - For each employee, count the number of Evidential Pricing Uploads specifically for 'JSW Brand' and 'Benchmark Brand'.
      - Group the data by 'State' and 'Employee Name', and aggregate the count of Evidential Pricing Uploads for each employee.

    - **Step 3: Identify Top 3 and Bottom 3 Employees**
      - For each state, sort employees by the count of Evidential Pricing Uploads in descending order to identify the Top 3 employees.
      - Sort employees in ascending order to identify the Bottom 3 employees.

    - **Step 4: Prepare and Display the Final Table**
      - The final dataframe should display the Top 3 and Bottom 3 employees state-wise with their respective counts of Evidential Pricing Uploads.
      - Include the following columns:
        - State
        - Employee Name
        - Upload Count
        - Rank (Top 3 or Bottom 3)
        Output should be in dataframe.Must follow {user_query}
    """
    response = smart_df.chat(prompt)
    return response


def handle_general_query(smart_df, user_query):
    general_query_prompt = f"""
    Answer the following user query: "{user_query}" using the dataframe and provide a detailed and accurate response.

    """
    response = smart_df.chat(general_query_prompt)
    return response

# Function to summarize dataframe response using OpenAI
def summarize_with_openai(text, user_query):
    client =OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": f"""
1] Comparison of NOP VS JSW BRAND WSP:
   - Consider NOP Only for JSW Brand.
   - Now Compare NOP With JSW Brand WSP:-
     - If NOP is greater than JSW Brand WSP upto 0 to 5 rs it is acceptable if it is more than you should rasied the flag along with date and district.
     - If NOP == WSP it is okay don't take any action.
     - If WSP is less than NOP then you need to raised the flag immediately.
                   
2] Comparison of JSW Brand WSP  VS Benchmark Brand WSP:
    - Compare JSW Brand WSP VS Benchmark Brand WSP.
    - If JSW Brand WSP is higher than Benchmark Brand WSP then it is okay but if it is less than then you should raised the flag like we can increase our price upto difference between Benchmark Brand WSP and JSW Brand WSP.
    
3] Provide the Summary District Wise:-
 - Provide NOP VS JSW Brand WSP Summary 
 - Provide JSW Brand WSP VS Bechmark Brand WSP Summary.

Don't Provide single above prompt in the summary just behave like above summary and follow the {user_query}.
Don't Provide criteria or condition in the summary.

Now you have to provide the answer of {user_query} while Considering the JSW Brand WSP vs Benchmark Brand WSP. if data is not available then don't provide the data.
Additional Price Support means (drop down request) it should be approved when JSW Brand WSP is grater than Benchmark Brand WSP.
Dataframe:
{text}
"""
}],
        max_tokens=4096
    )
    summary = completion.choices[0].message.content
    return summary


def summarize_with_top3(text, user_query):
    client =OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[{"role": "user", "content": f"""
Summarize the following dataframe with a focus on identifying price increase opportunities for the JSW Brand. 
Specifically, if the Benchmark Brand's price is higher than the JSW Brand's price, suggest how much we could increase our price to approach or match the Benchmark Brand's price.
AT the end you have to provide the decision of {user_query}.
                   
Dataframe:
{text}
"""
}],
        max_tokens=4096
    )
    summary = completion.choices[0].message.content
    return summary

def summarize_with_gener(text, user_query):
    client =OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model='gpt-4',
        messages=[{"role": "user", "content": f"""
Summarize the following dataframe.Give summary in breif and provide the decision of {user_query}.
Upload count columns should be the count of number must follow {user_query}.
Dataframe:
{text}
"""
}],
        max_tokens=4096
    )
    summary = completion.choices[0].message.content
    return summary


# Function to display plots
def display_plots():
    plots_dir = os.path.join(os.getcwd(), "plots")
    plot_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")), key=os.path.getmtime, reverse=True)
    
    if plot_files:
        for plot_file in plot_files:
            st.image(plot_file, caption=os.path.basename(plot_file), use_column_width=True)
    else:
        st.write("No plots available for display.")

def main():
    input_query = st.text_area("Ask your question here", placeholder="e.g., There is a request for additional price support from Uttarakhand Team wef 20th Apr, should the request be approved or rejected?")

    if input_query:
        if "additional price support" in input_query.lower():
            if st.button("Submit"):
                with st.spinner('Thinking....'):
                    response = get_answer(smart_df, input_query)
                    st.write(response)
                    summary = summarize_with_openai(response, input_query)
                    st.write(summary)
                   
        # Example usage in a Streamlit app
        elif "generate a day wise bar chart" in input_query.lower():
            if st.button("Submit"):
                with st.spinner('Loading plot...'):
                    display_plots()

        elif "show the top 3" in input_query.lower():
                if st.button("Submit"):
                    with st.spinner('Processing...'):
                        response = top3(smart_df, input_query)
                        st.write(response)
                        summary = summarize_with_top3(response, input_query)
                    st.write(summary)

        
        elif 'generate a list of top 3 and bottom 3' in input_query.lower():
            if st.button("Submit"):
                with st.spinner('Porcessing...'):
                    response=generate_l(smart_df,input_query)
                    st.write(response)
                    summary=summarize_with_gener(response,input_query)
                    st.write(summary)

        else:
            if st.button("Submit"):
                with st.spinner('Processing...'):
                    response = handle_general_query(smart_df, input_query)
                    st.write(response)

if __name__ == "__main__":
    main()
