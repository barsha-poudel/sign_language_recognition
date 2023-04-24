from flask import Flask,render_template
app = Flask(__name__)
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('index.html')
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()