from konlpy.tag import Mecab
mecab = Mecab(dicpath="C:/mecab/mecab-ko-dic")  # 실제 경로로 변경
print(mecab.pos("테스트"))