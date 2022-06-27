from tkinter import *
from enum import Enum
from model_development.fitting import fit_model, load_model
from model_development.my_model import create_model
from model_development.data_managing import create_train_vali_gens
from prediction_app import start_app

ready_model_path = 'archive/model/ready.json'
ready_weights_path = 'archive/model/ready.h5'

class Page(Enum):
    FIRST = 1
    FIT = 2
    TEST = 3

model = None
def create_page(root, page_func):
    for widget in root.winfo_children():
        widget.destroy()
    
    page = Frame(root)
    page.grid()

    page_func(root, page)

    for child in page.winfo_children():
        child.grid_configure(pady = 7.5, padx=7.5)

def get_change_page(root, page_to : Page):
    def changepage():
        for widget in root.winfo_children():
            widget.destroy()

        if page_to == Page.FIRST:
            create_page(root, first_page)

        elif page_to == Page.FIT:
            create_page(root, fit_page)

        elif page_to == Page.TEST:
            create_page(root, test_page)

    return changepage

def first_page(root, page):
    Label(page, justify='left', bg='#A4CDDA', text='''This is a GUI for the scripts I wrote,
This GUI will fit the model on the dataset.
In the final page, you will be able to launch the app,
which makes predictions using the model.

It IS posible to launch the app with a 
pretrained model at any step of the GUI.
When in the app, press q or escape in order to leave it.

*The GUI will be stuck whenever heavy computations occur.''')

    Button(page, text='Move To Fitting', command=get_change_page(root, Page.FIT), bg='#89E289')

    Button(page, text='Launch App (Pretrained Model)', command=launch_app, bg='#F26D70')

def fit_page(root, page):

    Label(page, justify='left', bg='#A4CDDA', text='''It IS posible to launch the app with a 
pretrained model at any step of the GUI.
When in the app, press q or escape in order to leave it.

*The GUI will be stuck whenever heavy computations occur.

Press START FITTING in order to start fitting the model.''')
    def fit():
        print('starting fitting')
        global model
        model = create_model()
        data = create_train_vali_gens(batch_size=64)
        fit_model(model, data, epochs=25, extra_callbacks=[])

    Button(page, text='START FITTING', command=fit, bg='#82C4E4')

    Button(page, text='Move To Test', command=get_change_page(root, Page.TEST), bg='#89E289')

    Button(page, text='Launch App (Pretrained Model)', command=launch_app, bg='#F26D70')

def test_page(root, page):

    Label(page, justify='left', bg='#A4CDDA', text='''*The GUI will be stuck whenever heavy computations occur.
When in the app, press q or escape in order to leave it.

Press START TESTING in order to start testing the model.

The results of the testing will be shown here''')

    res_label = Label(page, justify='left', bg='#A4CDDA', text='''Accuracy:
    Loss:''')
    
    def test():
        print('starting testing')
        data = create_train_vali_gens()[1] # in order to get the Validation data generator
        loss, acc = model.evaluate(data)
        loss_fix = str(loss)[:4]
        acc_fix_precentage = str(acc*100)[:4]
        res_label['text'] = f'''Accuracy: {acc_fix_precentage}%
Loss: {loss_fix}'''

    Button(page, text='START TESTING', command=test, bg='#82C4E4')
    
    Button(page, text='Launch App (With Currently Trained Model)', command=launch_app, bg='#89E289')

def launch_app():
    global model
    if model is None:
        model = load_model(ready_model_path, ready_weights_path)
    start_app(model)

if __name__ == '__main__':
    root = Tk()
    root.grid_columnconfigure(0, weight=1)
    create_page(root, first_page)
    root.title('Odelia')
    root.mainloop()