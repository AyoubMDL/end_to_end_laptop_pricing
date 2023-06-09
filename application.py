import logging
from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                brand=request.form['brand'],
                processor_brand=request.form['processor_brand'],
                processor_name=request.form['processor_name'],
                processor_gnrtn=request.form['processor_gnrtn'],
                ram_gb=request.form['ram_gb'],
                ram_type=request.form['ram_type'],
                ssd=request.form['ssd'],
                hdd=request.form['hdd'],
                os=request.form['os'],
                os_bit=request.form['os_bit'],
                graphic_card_gb=request.form['graphic_card_gb'],
                weight=request.form['weight'],
                warranty=request.form['warranty'],
                touchscreen=request.form['Touchscreen'],
                msoffice=request.form['msoffice'],
                rating=request.form['rating']
            )

            pred_df = data.get_data_as_data_frame()
            prediction_pipeline = PredictPipeline()
            prediction = prediction_pipeline.predict(pred_df)

            return render_template("home.html", results=round(prediction[0], 2))

        except Exception as e:
            # Log the exception with detailed message
            logging.exception("An error occurred during request processing: %s", str(e))
            # Render an error page or display a user-friendly error message
            return render_template("error.html", error_message="An internal server error occurred: {}".format(str(e)))

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8080)
