import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
import pdfplumber

# Constants
MAX_TOKENS = 4096
TEMPERATURE = 0.7

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Streamlit app
def main():
    st.title("Invoice PDF Parser")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text from the PDF
        text = extract_text_from_pdf("temp_file.pdf")
        # processed_text = " ".join(text.split("\n"))
        processed_text = text
        # check if the text is extracted is not empty
        if not processed_text:
            st.error("No text extracted from PDF. Please upload a different PDF file.")
            return  
        with st.expander("Extracted text from PDF"):
            st.write(processed_text)

        # Initialize the GPT4 model
        _ = load_dotenv(find_dotenv())
        llm = AzureChatOpenAI(
            openai_api_type=os.environ['OPENAI_API_TYPE'],
            openai_api_base=os.environ['OPENAI_API_BASE'],
            openai_api_version=os.environ['OPENAI_API_VERSION'],
            deployment_name=os.environ['OPENAI_API_DEPLOYMENT'],
            model_name=os.environ['OPENAI_API_MODEL'],
            openai_api_key=os.environ['OPENAI_API_KEY'],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        invoice_schema = Object(
            id="invoice_extraction",
            description="extraction of relevant information from invoice",
            attributes=[
                Text(
                    id="invoice_number",
                    description="unique number (identifier) of given invoice",
                    examples=[
                        ("1019273", "1019273"),
                        ("291870.4-1", "291870.4-1"),
                        ("291870.5-1", "291870.5-1")
                    ]
                ),
                Text(
                    id="invoice_date",
                    description="date of the invoice",
                    examples=[
                        ("12/20/2023", "12/20/2023"),
                        ("2024-05-27", "2024-05-27")
                    ]
                ),
                Text(
                    id="customer_id",
                    description="identifier for the customer",
                    examples=[
                        ("47900", "47900"),
                        ("XT5570", "XT5570")
                    ]
                ),
                Text(
                    id="customer_po",
                    description="customer purchase order number",
                    examples=[
                        ("832970", "832970"),
                        ("838290", "838290"),
                        ("838940", "838940")
                    ]
                ),
                Text(
                    id="salesperson",
                    description="salesperson handling the invoice",
                    examples=[
                        ("James Flemming", "James Flemming"),
                        ("Elaine Kirby", "Elaine Kirby")
                    ]
                ),
                Text(
                    id="terms",
                    description="payment terms",
                    examples=[
                        ("Net 30", "Net 30"),
                        ("NET 45", "NET 45")
                    ]
                ),
                Text(
                    id="due_date",
                    description="due date for payment",
                    examples=[
                        ("1/19/2024", "1/19/2024"),
                        ("Jul 12, 2024", "Jul 12, 2024")
                    ]
                ),
                Text(
                    id="packing_slip_no",
                    description="packing slip number",
                    examples=[
                        ("20369", "20369")
                    ]
                ),
                Text(
                    id="tracking_number",
                    description="tracking number for the shipment",
                    examples=[
                        ("1Z1R7W540343315285", "1Z1R7W540343315285"),
                        ("1Z8568830390831159", "1Z8568830390831159"),
                        ("1Z8868940261123687", "1Z8868940261123687"),
                        ("1Z8568830391553949", "1Z8568830391553949")
                    ]
                )
            ],
            many=False,
        )

        billing_address_schema = Object(
            id="billing_address",
            description="where the bill for a product or service is sent so it can be paid by the recipient, also known as Bill to or Authorized by",
            attributes=[
                Text(id="name", description="the name of person and organization"),
                Text(id="address_line", description="the local delivery information such as street, building number, PO box, or apartment portion of a postal address"),
                Text(id="city", description="the city portion of the address"),
                Text(id="state_province_code", description="the code for address US states"),
                Text(id="postal_code", description="the postal code portion of the address"),
                Text(id="country", description="the country portion of the address")
            ],
            examples=[
                (
                    "B I L L T O TRIMBLE NAVIGATION LTD - SUNNYVALE 5475 KELLENBURGER ROAD DAYTON OH 45424 United States of America",
                    {
                        "name": "TRIMBLE NAVIGATION LTD - SUNNYVALE",
                        "address_line": "5475 KELLENBURGER ROAD",
                        "city": "DAYTON",
                        "state_province_code": "OH",
                        "postal_code": "45424",
                        "country": "United States of America"
                    }
                ),
                (
                    "Bill To Trimble Navigation Sunnyvale & Dayton AP 5475 Kellenburger Road Dayton OH 45424-1099 USA",
                    {
                        "name": "Trimble Navigation Sunnyvale & Dayton AP",
                        "address_line": "5475 Kellenburger Road",
                        "city": "Dayton",
                        "state_province_code": "OH",
                        "postal_code": "45424-1099",
                        "country": "USA"
                    }
                )
            ],
            many=False,  # This indicates multiple billing addresses can be extracted
        )


        remit_address_schema = Object(
            id="remit_address",
            description="where the product or service is remit to",
            attributes=[
                Text(id="name", description="the name of person and organization"),
                Text(id="address_line", description="the local delivery information such as street, building number, PO box, or apartment portion of a postal address"),
                Text(id="city", description="the city portion of the address"),
                Text(id="state_province_code", description="the code for address US states"),
                Text(id="postal_code", description="the postal code portion of the address"),
                Text(id="country", description="the country portion of the address")
            ],
            examples=[
                (
                    "Remit To Amphenol DC Electronics 28704 Network Place Chicago, IL 60673-1286",
                    {
                        "name": "Amphenol DC Electronics",
                        "address_line": "28704 Network Place",
                        "city": "Chicago",
                        "state_province_code": "IL",
                        "postal_code": "60673-1286",
                        "country": "USA"
                    }
                ),
                (
                    "TRIMBLE NAVIGATION LTD 4450 GIBSON DRIVE TIPP CITY OH 45371 United States of America Phone: 650-593-3288 Remit To: ",
                    {
                        "name": "TRIMBLE NAVIGATION LTD",
                        "address_line": "4450 GIBSON DRIVE",
                        "city": "TIPP CITY",
                        "state_province_code": "OH",
                        "postal_code": "45371",
                        "country": "United States of America"
                    }
                ),
                (
                    "Remit To: TRIMBLE NAVIGATION LTD 4450 GIBSON DRIVE TIPP CITY OH 45371 United States of America",
                    {
                        "name": "TRIMBLE NAVIGATION LTD",
                        "address_line": "4450 GIBSON DRIVE",
                        "city": "TIPP CITY",
                        "state_province_code": "OH",
                        "postal_code": "45371",
                        "country": "United States of America"
                    }
                )
            ],
            many=True  # Only one remit address should be extracted
        )


        shipping_address_schema = Object(
            id="shipping_address",
            description="where the product or service is shipped",
            attributes=[
                Text(id="name", description="the name of person and organization"),
                Text(id="address_line", description="the local delivery information such as street, building number, PO box, or apartment portion of a postal address"),
                Text(id="city", description="the city portion of the address"),
                Text(id="state_province_code", description="the code for address US states"),
                Text(id="postal_code", description="the postal code portion of the address"),
                Text(id="country", description="the country portion of the address")
            ],
            examples=[
                (
                    "SHIP TO TRIMBLE NAVIGATION LTD 4450 GIBSON DRIVE TIPP CITY OH 45371 United States of America",
                    {
                        "name": "TRIMBLE NAVIGATION LTD",
                        "address_line": "4450 GIBSON DRIVE",
                        "city": "TIPP CITY",
                        "state_province_code": "OH",
                        "postal_code": "45371",
                        "country": "United States of America"
                    }
                ),
                (
                    "Ship To Amphenol DC Electronics 28704 Network Place Chicago, IL 60673-1286",
                    {
                        "name": "Amphenol DC Electronics",
                        "address_line": "28704 Network Place",
                        "city": "Chicago",
                        "state_province_code": "IL",
                        "postal_code": "60673-1286",
                        "country": "USA"
                    }
                )
            ],
            many=False,  # This indicates multiple shipping addresses can be extracted
        )

        products_schema = Object(
            id="bill",
            description="the details of bill",
            attributes=[
                Text(id="product_description", description="the description of the product or service"),
                Text(id="count", description="number of units bought for the product"),
                Text(id="unit_item_price", description="price per unit"),
                Text(id="product_total_price", description="the total price, which is number of units * unit_price"),
            ],
            examples=[
                (
                    "SC8702 3/8-16X2 HX-CAP-SC-GR8-Z-YEL - 132091 100 0.90000 90.00",
                    {
                        "product_description": "SC8702 3/8-16X2 HX-CAP-SC-GR8-Z-YEL - 132091",
                        "count": 100,
                        "unit_item_price": 0.90000,
                        "product_total_price": 90.00,
                    }
                ),
                (
                    "CABLE, GNSS ANTENNA, 20 M 40 $197.58 $7,903.20",
                    {
                        "product_description": "CABLE, GNSS ANTENNA, 20 M",
                        "count": 40,
                        "unit_item_price": 197.58,
                        "product_total_price": 7903.20,
                    }
                ),
                (
                    "CABLE, RADIO ANTENNA, UHF, TNC TO N, LMR400-UF 10 $90.93 $909.30",
                    {
                        "product_description": "CABLE, RADIO ANTENNA, UHF, TNC TO N, LMR400-UF",
                        "count": 10,
                        "unit_item_price": 90.93,
                        "product_total_price": 909.30,
                    }
                ),
                (
                    "POLE,EXTENSION,1ft,TNL REV A 50 15.91000 795.50",
                    {
                        "product_description": "POLE,EXTENSION,1ft,TNL REV A",
                        "count": 50,
                        "unit_item_price": 15.91,
                        "product_total_price": 795.50,
                    }
                )
            ],
            many=True  # This indicates multiple products can be extracted
        )


        # Extract the invoice number
        invoice_chain = create_extraction_chain(llm, invoice_schema)
        invoice = invoice_chain.predict_and_parse(text=processed_text)["data"]
        st.subheader("Invoice Details")
        st.json(invoice)
        
        # Extract the Remit address
        remit_address_chain = create_extraction_chain(llm, remit_address_schema)
        remit_address = remit_address_chain.predict_and_parse(text=processed_text)["data"]
        st.subheader("Remit To Address Details")
        st.json(remit_address)
        
        # Extract the billing address
        billing_address_chain = create_extraction_chain(llm, billing_address_schema)
        billing_address = billing_address_chain.predict_and_parse(text=processed_text)["data"]
        st.subheader("Billing Address Details")
        st.json(billing_address['billing_address'][0])
        
        # Extract the shipping address
        shipping_address_chain = create_extraction_chain(llm, shipping_address_schema)
        shipping_address = shipping_address_chain.predict_and_parse(text=processed_text)["data"]
        st.subheader("Shipping Address Details")
        st.json(shipping_address['shipping_address'][0])

        # Extract the product address
        products_chain = create_extraction_chain(llm, products_schema)
        products = products_chain.predict_and_parse(text=processed_text)["data"]
        st.subheader("Product Details")
        st.json(products)

        # Clean up the temporary file
        os.remove("temp_file.pdf")

# Run the Streamlit app
if __name__ == "__main__":
    main()