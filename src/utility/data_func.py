class CancerTypeDic:

    """
    CRC = ColonRectum

    IN TEP DATASETS:
    Chol: 胆管癌   GBM: 胶质母细胞瘤(颅内)
    NSCLC is a popular kind of Lung Cancer

    IN HUST DATASETS:
    Liver and Pancreatic are combined into one category;
    Urinary, Thyroid are new categories;
    """

    def __init__(self):
        self.type_dict_GSE = {
                "Health": 0, "Breast": 1, "Liver": 2, "Chol": 3, "CRC": 4, "GBM": 5,
                "Platelet": 6, "Lung": 7, "Panc": 8, "NSCLC": 9, "Urinary": 10, "Thyroid":11,
                "LiverAndPancreatic": 12, "Stomach": 13,"Unknown": -1
        }
        self.type_dict_Hust = {
            "Health": 0, "Breast": 1, "CRC": 2, "Lung": 3, "Thyroid": 4,
            "LiverAndPancreatic": 5, "Stomach": 6, "Urinary": 7, "Unknown": -1
        }

        self.reversed_type_dict_GSE = {}
        self.reversed_type_dict_Hust = {}

        for key, val in self.type_dict_GSE.items():
            self.reversed_type_dict_GSE[val] = key
        for key, val in self.type_dict_Hust.items():
            self.reversed_type_dict_Hust[val] = key

    def TypetoID(self, type_name, dict_name):
        if dict_name == "Hust":
            if type_name in self.type_dict_Hust:
                return self.type_dict_Hust[type_name]
            else:
                print("Not Found!")
                print(self.type_dict_Hust.index)
                return -999
        elif dict_name == "GSE":
            if type_name in self.type_dict_GSE:
                return self.type_dict_GSE[type_name]
            else:
                print("Not Found!")
                print(self.type_dict_GSE.index)
                return -999
        else:
            raise NotImplementedError


    def IDtoType(self, id, dict_name):
        if dict_name == "Hust":
            if id in self.reversed_type_dict_Hust:
                return self.reversed_type_dict_Hust[id]
            else:
                print("Not Found!")
                print(self.reversed_type_dict_Hust.index)
                return "NotFound"
        elif dict_name == "GSE":
            if id in self.reversed_type_dict_GSE:
                return self.reversed_type_dict_GSE[id]
            else:
                print("Not Found!")
                print(self.reversed_type_dict_GSE.index)
                return "NotFound"
        else:
            raise NotImplementedError


