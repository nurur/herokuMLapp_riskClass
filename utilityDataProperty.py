import streamlit as st


def get_dataProperty():
	
	st.info("Data Properties")

	text = """
	<p style="font-family:Serif; color:Black; font-size: 18px;">
	Response: Customer risk profile: 1) low risk, and 
	2) high risk. 
	</p>

	<p style="font-family:Serif; color:Black; font-size: 18px;">
	Predictor: There dataset has a total of nine predictors. 

	<p style="font-family:Serif; color:Black; font-size: 18px;">
	It has the following five categorical features:<br>
	1) Gender,
	2) Mortgage,
	3) Marital Status,
	4) Number of Loans, and 
	5) Education Level. </p>

	<p style="font-family:Serif; color:Black; font-size: 18px;">
	It contains the following three numerical features:<br>
	1) Age,
	2) Income, and 
	3) Years of Work Experience. </p>
	"""

	st.markdown(text, unsafe_allow_html=True)

