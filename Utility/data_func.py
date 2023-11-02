class CancerTypeDic:
    def __init__(self):
        self.type_dict = {"Health": 0, "Breast": 1, "Liver": 2, "Chol": 3, "CRC": 4, "GBM": 5,
                     "Platelet": 6, "Lung": 7, "Panc": 8, "NSCLC": 9, "Unknown": -1}
        self.revesed_type_dict = {}
        for key, val in self.type_dict.items():
            self.revesed_type_dict[val] = key

    def TypetoID(self, type_name):
        if type_name in self.type_dict:
            return self.type_dict[type_name]
        else:
            print("Not Found!")
            print(self.type_dict.index)
            return -999

    def IDtoType(self, id):
        if id in self.revesed_type_dict:
            return self.revesed_type_dict[id]
        else:
            print("Not Found!")
            print(self.revesed_type_dict.index)
            return "NotFound"
