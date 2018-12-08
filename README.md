Python Memo : 파이썬 관련 메모
===

#### Python code written while studying
* Python 코드 작성 중 자주 검색해보는 것들 또는 까먹지 말아야 할 것 메모했음.
* 전처리, 사용자 함수, 자주 사용하는 코드, 이론 등


*이론*
===

튜플
---
* 튜플은 리스트 [a,b, ...]와 비슷하다. 차이점은 튜플은 요소를 추가, 변경, 삭제하는 처리가 불가능하고 리스트는 가능하다는 것이다. 튜플이 메모리를 적게 사용하기 때문에 요소를 수정할 필요가 없으면 튜플을 사용하는 것이 효율적이다.










_전처리_
===


* 1.공백 제거한 뒤 변수명 일괄 변경
```
' '.join(k).split()

ex)
shares = pd.read_csv("OnlineNewsPopularity.csv")
k=shares.columns.values.tolist()
k=' '.join(k).split()
shares.columns=' '.join(shares.columns.values).split()
```

* 2.열 선택
    * a에 해당하는 열 가져오기
    ```
    a=["a","b","c"]
    shares[a] #a,b,c열 가져와짐.
    ```
    * a가 아닌 열 가져오기
    ```
    shares[shares.columns.difference(a)]
    ```
    * 열 붙히기(cbind)
    ```
    df_c = pd.concat([k1, k2], axis=1)
    ```
<br />

_함수_
===
* len() : 문자열, 리스트 길이 반환.
* iloc() : DataFrame 함수로 특정 행의 데이터를 가져온다.
* int() : 정수로 형식 변환.
    ```
    int(0.5) = 1       int(1.3)=1
    ```
* np.random.rand() : Numpy의 random 모듈은 랜덤 값 생성을 위한 rand() 함수를 제공함.
0에서 1사이의 값을 생성하여 반환함. randint(low, high=None) 함수는 high를 넣지 않은 경우에는 0에서 low사이의 정수를 랜덤으로 생성하고 high를 넣은 경우 low에서 high 사이의 정수를 생성함.

* np.argmax() : argmax(array) 함수는 입력으로 들어온 array에서 가장 큰 값의 위치를 반환한다.
    ```
    예를 들어 array = [3,5,7,0,-3] dlaus 가장 큰 값은 7이고 그 위치는 2이다. 
    np.argmax(array) => 2
    ```
* Numpy의 reshpe() 함수
    * ndarray를 reshape() 함수를 써서 다른 차원으로 변환할 수 있다. 예를 들어 1차원 배열인 [1,2,3,4,5,6]이 있을 때 이 배열의 shape(즉,모양)은 (6,)이다. 이를 (3,2)로 만들면 [ [1,2],[3,4],[5,6] ]이 된다. 이 때 유의할 점은 배열의 총 크기는 변하지 않아야 한다. 이 예에서는 변환한 후의 크기가 3x2=6 이므로 1차원 배열일 때와 같다.
* Numpy의 arrange() 함수
    ```
    np.arrange(3) -> 배열 [0,1,2]를 반환. 
    시작값(start), 종료값(stop), 값 사이 간격(step) 지정 가능.
    ```
* Numpy의 zeros() 함수
    * 인자로 배열의 형태인 shape을 받아서 0으로 구성된 NumPy 배열을 반환한다.
    ```
    ex) zeros(3) -> [0,0,0] 을 반환,
        zeros((2,2)) -> [ [0,0], [0,0]] 을 반환.
    ```
    * 주의할 점은 다차원 배열의 경우 그 형태를 튜플로 넘겨줘야 한다는 점이다.
* Matplotlib의 subplots() 함수
    * 여러 Axes로 구성된 Figure를 생성할 때 효과적이다. 인자로 들어가는 nrows는 행 개수,ncols는 열 개수를 의미한다. nrow가 2이고 ncol이 3이라면 2x3 Axes, 즉 6개의 Axes로 구성된 Figure가 생성된다.
* zip() 함수
    * 파이썬 내장 함수로 두 개의 배열에서 같은 인덱스의 요소를 순서대로 묶어준다.
    ```
    zip([1,2,3],[4,5,6]) -> [(1,4),(2,5),(3,6)] 
    ```
* Matplotlib()의 axvline() 함수
    * x축 위치에서 세로로 선을 긋는 함수이다. 이선의 색깔은 color 인자로, 선의 투명도를 alpha로 정해줄 수 있다.
* Matplotlib()의 plot() 함수
    * x축의 데이터, y축의 데이터, 차트의 스타일을 인자로 받는다. x축 데이터와 y축 데이터의 크기가 같아야 한다. 스타일은 실선, 점선 등의 선 형태와 선의 색깔을 축약적으로 정의한다.
    * 표시할 다양한 모양과 색깔을 좆합할 수 있다.
    ```
    ex1) axes[1].plot(x, num_stocks,'-k')에서 '-k' 는 검정색 실선을 의미함.
    ex2) '-'은 선, '.'은 점, 'r'은 빨간색, 'b'는 파란색을 의미.
         이를 조합한 '-r','-b' 등은 빨간 선, 파란 선을 의미.
    ```
* Matplotlib()의 fill_between() 함수
    * x축 배열과 두 개의 y축 배열을 입력으로 받는다.
    두 개의 y축 배열의 같은 인덱스 위치의 값 사이에 색을 치란다.
    where 옵션으로 색을 칠할 조건을 추가할 수 있고 facecolor 옵션으로 칠할 색을 지정하고
    alpha 옵션으로 투명도를 조정할 수 있다.
* Matplotlib()의 tight_layout() 함수
    * Fiqure의 크기에 알맞게 내부 차트들의 크기를 조정해 준다.
<br />

_ETC_
===
* ` __init__() `
    * 파이썬 클래스의 생성사 __init__()은 클래스의 객체가 생성될 때 자동으로 호출되는 함수임. 보통 이 함수에서 입력으로 받은 값들을 객체 내의 변수로 할당해 준다.
* 문자열에 값을 넣어주는 방법 `%d, %s, %f` and `format()`
    * `%d`는 정수, `%s`는 문자열, `%f`는 실수를 입력하는 자리이다.
    *   ```
        ex1) ['Hello %s' % 'World!']는 ['Hello World']가 된다.
        ex2) ['%d / %d = %.2f' % (10, 3, 10/3)] 이 [ '10 / 3 = 3.33' ]으로 된다.
        ```
    * 다른 방법으로 `format()` 함수가 있다.
    이 함수에는 값을 넣을 자리를 {}로 표현한다.
    ```
    ex1) ['Hello {}'.format('World!')] -> ['Hello World!']
    ex2) ['Hello {tail}'.format(tail='World!')] -> ['Hello World!']
    ``` 
