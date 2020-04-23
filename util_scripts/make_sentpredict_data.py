import sys
import os

lan = "fra"
dia = "XXfr-ca"
data_dir = "data/dialects/fra/"

for split in ["train", "dev", "test"]:
    label_out = open(data_dir + "prediction16k/" + split + ".label", "w")
    input_out = open(data_dir + "prediction16k/" + split + ".input0", "w")

    lan_file = open(data_dir + "{}_eng/ted-{}.orig.spm16000.{}".format(lan, split, lan), "r")
    dia_file = open(data_dir + "{}_eng/ted-{}.orig.spm16000.{}".format(dia, split, dia), "r")
    lan_lines = lan_file.readlines()
    dia_lines = dia_file.readlines()
    if split == "train":
        for i in range(len(lan_lines)):
            input_out.write(lan_lines[i])
            label_out.write("main_language\n")
            input_out.write(dia_lines[i%len(dia_lines)])
            label_out.write("dialect\n")
    else:
        for i in range(len(lan_lines)):
            input_out.write(lan_lines[i])
            label_out.write("main_language\n")
        for i in range(len(dia_lines)):
            input_out.write(dia_lines[i])
            label_out.write("dialect\n")
