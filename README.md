## Configuration

Create a file named `.env` in the root directory of your project.

Add the following environment variables to the `.env` file

## Running the Application

Open a terminal in the project directory and run the following command:

```bash
streamlit run TrimbleInvoiceParser.py

## Usage

1. Upload a PDF invoice using the file uploader.
2. The application will extract the following information from the invoice:
   - Invoice Details (invoice number, date, etc.)
   - Remit To Address
   - Billing Address
   - Shipping Address
   - Product Details (description, quantity, price, etc.)

## How it Works

1. The application extracts text from the uploaded PDF using `pdfplumber`.
2. It then uses a pre-trained GPT-4 model (loaded via `langchain` and `kor` libraries) to identify and extract relevant information based on predefined schemas.
3. Finally, the extracted information is displayed in the Streamlit app.

### Note

- This is a basic example and may require further customization depending on the specific format of your invoices.
- The accuracy of the extraction depends on the quality of the training data used for the GPT-4 model.
