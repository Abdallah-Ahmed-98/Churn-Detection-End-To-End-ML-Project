from flask import Flask, render_template, request
import os 
import pandas as pd
from mlProject import logger
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 



@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            credit_score =int(request.form['credit_score'])
            geography =request.form['geography']
            gender =request.form['gender']
            age =int(request.form['age'])
            tenure =int(request.form['tenure'])
            balance =float(request.form['balance'])
            num_of_products =int(request.form['num_of_products'])
            has_cr_card =int(request.form['has_cr_card'])
            is_active_member =int(request.form['is_active_member'])
            estimated_salary =float(request.form['estimated_salary'])
       
            data_headers = ["RowNumber",
                            "CustomerId",
                            "Surname",
                            "CreditScore",
                            "Geography",
                            "Gender",
                            "Age",
                            "Tenure",
                            "Balance",
                            "NumOfProducts",
                            "HasCrCard",
                            "IsActiveMember",
                            "EstimatedSalary"]
         
            data_values = [1,
                           1,
                           "no name",
                            credit_score,
                            geography,
                            gender,
                            age,
                            tenure,
                            balance,
                            num_of_products,
                            has_cr_card,
                            is_active_member,
                            estimated_salary]
            

            data = pd.DataFrame([data_values], columns=data_headers)

####################################################################### Prediction stage #######################################################################

            STAGE_NAME = "Prediction stage"
            try:
                logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
                prediction = PredictionPipeline()
                predict = prediction.main(data)
                if predict == 1:
                    result = "Exited"
                else:
                    result = "Not Exited"                
                logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            except Exception as e:
                logger.exception(e)
                raise e
            
 ################################################################################################################################################################

            return render_template('results.html', prediction = result)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')
    



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080, debug=True)