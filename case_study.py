import torch
from case_study.knowledge_graph import Knowledge_Graph
import os
import json
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
                             QFormLayout, QLineEdit, QPushButton,
                             QSpinBox, QCompleter, QTableView, QGridLayout, QGroupBox, QHeaderView)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from case_study.pandas_qt import DataFrameModel

import sys

import pandas as pd


class App(QDialog):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.resize(1203, 460)
        self.setWindowTitle('Link Predition App')
        self.create_form()

        df = pd.DataFrame(columns=['HEAD', 'RELATION', 'TAIL', 'SCORE', 'RANK', 'EXISTING_IN_TEST_DATASET'])
        self.table_view = QTableView()
        self.table_view.setModel(DataFrameModel(df))
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.form_group_box, 0, 0)
        self.main_layout.addWidget(self.table_view, 0, 1)
        self.setLayout(self.main_layout)
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 5)

    def create_form(self):
        self.form_group_box = QGroupBox()
        with open('case_study/demo.json') as demo_f:
            self.demo_dict = json.load(demo_f)
        self.load_model()
        entity_list = list(self.demo_dict.keys())
        entity_completer = QCompleter(entity_list)
        self.entity = QLineEdit()
        self.entity.setCompleter(entity_completer)

        self.relation = QComboBox()

        self.predict_type = QComboBox()
        self.predict_type.addItems(['Predict Tail Entity', 'Predict Head Entity'])

        self.predict_techique = QComboBox()
        self.predict_techique.addItems(['Top N Highest Score', 'Threshold'])
        self.predict_techique.currentIndexChanged.connect(self.selection_change)

        self.n = QSpinBox(self)
        self.n.setValue(10)

        self.threshold = None

        self.button_predict = QPushButton('Predict', self)
        self.button_predict.clicked.connect(self.handle_predict)

        self.entity.editingFinished.connect(self.update_relation)

        self.form = QFormLayout()
        self.form.addRow('Enity', self.entity)
        self.form.addRow('Relation', self.relation)
        self.form.addRow('Predict Type', self.predict_type)
        self.form.addRow('Predict Techique', self.predict_techique)
        self.form.addRow('N', self.n)
        self.form.addWidget(self.button_predict)
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form_group_box.setLayout(self.form)

    def update_relation(self):
        ent = self.entity.text()
        try:
            rel_list = self.demo_dict[ent]['relation']
            self.relation.clear()
            self.relation.addItems(rel_list)
            self.relation.update()
        except:
            pass

    def selection_change(self):
        if self.predict_techique.currentText() == 'Top N Highest Score':
            self.form.removeRow(self.threshold)
            self.form.removeWidget(self.button_predict)
            self.threshold = None
            self.n = QSpinBox(self)
            self.n.setValue(10)
            self.form.addRow('N', self.n)
            self.form.addWidget(self.button_predict)
        else:
            self.form.removeRow(self.n)
            self.form.removeWidget(self.button_predict)
            self.n = None
            self.threshold = QLineEdit(self)
            self.threshold.setValidator(QDoubleValidator(0.0, 1.0, 2))
            self.threshold.setText('0.5')
            self.form.addRow('Threshold', self.threshold)
            self.form.addWidget(self.button_predict)

    def handle_predict(self):
        try:
            ent = self.demo_dict[self.entity.text()]['id']
            rel = self.relation.currentText()
            tail_predict = True if self.predict_type.currentText() == 'Predict Tail Entity' else False
            n = int(self.n.text()) if self.n is not None else None
            threshold = float(self.threshold.text()) if self.threshold is not None else None

            result = self.kg.link_prediction(ent, rel, tail_predict, n, threshold)
            # print(result)
            self.main_layout.removeWidget(self.table_view)
            self.table_view = QTableView()
            header = self.table_view.horizontalHeader()
            self.table_view.setModel(DataFrameModel(result))
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
            self.main_layout.addWidget(self.table_view, 0, 1)
        except:
            pass

    def load_model(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            device = torch.device('cpu')
        pretrained_path = os.path.join('checkpoints', 'compgat_conve.pth')
        model = torch.load(pretrained_path)
        model.to(device)
        self.kg = Knowledge_Graph('FB15k-237', model, device)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    link_predict = App()
    sys.exit(link_predict.exec_())
