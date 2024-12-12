from dataset import FileWork
from file_infrormation import Preparation_df
from apriori import Apriori
from eclat import EclatAlgorithm
from fp_growth import FP_Growth
def main():
    FileWork.work_with_file()
    df = Preparation_df.main()
    Apriori.apriori_algorithm()
    EclatAlgorithm.main()
    FP_Growth.main()
if __name__ == '__main__':
    main()

