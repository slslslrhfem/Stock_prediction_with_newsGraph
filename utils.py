import datetime

def check_outlier(lists): #0이나 99999... 같은 이상한 값 있으면 데이터 안씀. 주식이 실제로 150배가 올랐을 수도 있지만 걍 min*150이 max보다 커지면 없애는 식으로 구현
#원래 Nasdaq하려고 yahoo에서 데이터 긁어올때는 이거 필수였는데 코스피에서는 거의 안걸리는거 같긴 함. 그냥 있으니까 씀
  to_check_list=lists[:,1]
  to_check_list2=lists[:,2]
  minimum = min(to_check_list)
  maximum = max(to_check_list)
  minimum2 = min(to_check_list2)
  maximum2 = max(to_check_list2)
  if (maximum > 150* minimum or maximum2 > 150*minimum2 or minimum == maximum or minimum2 == maximum2):
    return False
  return True


def Label_changer(sector_label): # Sector가 192개(2023-05-27기준)쯤 되는데, 1Sector에 1개의 회사만 속한 경우도 있고 해서 "주관적으로" 비슷한 Sector끼리 묶어 Label을 재지정합니다. 애매한 케이스나, 분리해야할 것 같은 Sector가 있긴 합니다만.. 일단 모르겠습니다
  software = ['소프트웨어 개발 및 공급업','컴퓨터 프로그래밍, 시스템 통합 및 관리업',	 '자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업']
  mechanic = ['특수 목적용 기계 제조업',  '일반 목적용 기계 제조업', '전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업', '금속 주조업','증기, 냉·온수 및 공기조절 공급업',  '산업용 기계 및 장비 임대업', '측정, 시험, 항해, 제어 및 기타 정밀기기 제조업; 광학기기 제외',  '전구 및 조명장치 제조업','항공기,우주선 및 부품 제조업', '사진장비 및 광학기기 제조업']
  automobile = ['철도장비 제조업','자동차 차체 및 트레일러 제조업','자동차용 엔진 및 자동차 제조업','자동차 재제조 부품 제조업',  '자동차 신품 부품 제조업', 	 '선박 및 보트 건조업']
  electronic = [ '전자부품 제조업','절연선 및 케이블 제조업','일차전지 및 축전지 제조업','기타 전기장비 제조업','컴퓨터 및 주변장치 제조업','가정용 기기 제조업','전기업','반도체 제조업','통신 및 방송 장비 제조업','전기 통신업','전기 및 통신 공사업']
  medical = [ '의약품 제조업','기초 의약물질 및 생물학적 제제 제조업','의료용 기기 제조업','의료용품 및 기타 의약 관련제품 제조업']
  economy = [ '기타 금융업','보험 및 연금관련 서비스업', '신탁업 및 집합투자업', '재 보험업','금융 지원 서비스업','보험업', '은행 및 저축기관']
  chemistry = [ '기타 화학제품 제조업', '기초 화학물질 제조업','연료용 가스 제조 및 배관공급업','석유 정제품 제조업','화학섬유 제조업', '해체, 선별 및 원료 재생업']
  entertainment = [ '영화, 비디오물, 방송프로그램 제작 및 배급업', '운동 및 경기용구 제조업','스포츠 서비스업','광고업','창작 및 예술관련 서비스업','영상 및 음향기기 제조업','텔레비전 방송업','서적, 잡지 및 기타 인쇄물 출판업','오디오물 출판 및 원판 녹음업']
  science = [ '자연과학 및 공학 연구개발업', '그외 기타 전문, 과학 및 기술 서비스업', '기타 과학기술 서비스업']
  food = ['기타 식품 제조업', '비알코올음료 및 얼음 제조업', '작물 재배업', '알코올음료 제조업', '음식점업',	'과실, 채소 가공 및 저장 처리업','곡물가공품, 전분 및 전분제품 제조업','음·식료품 및 담배 소매업','도축, 육류 가공 및 저장 처리업','낙농제품 및 식용빙과류 제조업','수산물 가공 및 저장 처리업','어로 어업','동물용 사료 및 조제식품 제조업']
  clothes = [ '봉제의복 제조업', '의복 액세서리 제조업','편조원단 제조업','기타 섬유제품 제조업', '편조의복 제조업', '섬유제품 염색, 정리 및 마무리 가공업', '직물직조 및 직물제품 제조업', '신발 및 신발 부분품 제조업', '귀금속 및 장신용품 제조업', '가죽, 가방 및 유사제품 제조업']
  transport = [ '도로 화물 운송업', '기타 운송관련 서비스업', '육상 여객 운송업', '항공 여객 운송업', '해상 운송업', '운송장비 임대업']
  construction = [ '부동산 임대 및 공급업', '건물 건설업', '토목 건설업', '건축기술, 엔지니어링 및 관련 기술 서비스업', '실내건축 및 건축마무리 공사업', '건물설비 설치 공사업', '기반조성 및 시설물 축조관련 전문공사업']
  wholesale_retail = [ '상품 종합 도매업', '기타 상품 전문 소매업', '기타 전문 도매업', '기타 생활용품 소매업', '기계장비 및 관련 물품 도매업', '건축자재, 철물 및 난방장치 도매업', '종합 소매업', '산업용 농·축산물 및 동·식물 도매업', '생활용품 도매업', '가전제품 및 정보통신장비 소매업', '섬유, 의복, 신발 및 가죽제품 소매업', '음·식료품 및 담배 도매업', '무점포 소매업', '자동차 부품 및 내장품 판매업', '자동차 판매업', '연료 소매업']
  other_service = ['전문디자인업', '기타 정보 서비스업', '교육지원 서비스업', '초등 교육기관', '상품 중개업', '기록매체 복제업', '일반 교습 학원', '인쇄 및 인쇄관련 산업', '여행사 및 기타 여행보조 서비스업', '기타 교육기관', '개인 및 가정용품 임대업','유원지 및 기타 오락관련 서비스업','시장조사 및 여론조사업', '폐기물 처리업','그외 기타 개인 서비스업','일반 및 생활 숙박시설 운영업','경비, 경호 및 탐정업', '기타 비금속광물 광업','사업시설 유지·관리 서비스업','기타 전문 서비스업','기타 사업지원 서비스업','회사 본부 및 경영 컨설팅 서비스업']
  other_manufacture = ['1차 철강 제조업', '제재 및 목재 가공업', '내화, 비내화 요업제품 제조업', '플라스틱제품 제조업', '유리 및 유리제품 제조업', '그외 기타 운송장비 제조업', '시멘트, 석회, 플라스터 및 그 제품 제조업', '골판지, 종이 상자 및 종이용기 제조업', '기타 종이 및 판지 제품 제조업', '나무제품 제조업', '기타 비금속 광물제품 제조업', '기타 금속 가공제품 제조업', '무기 및 총포탄 제조업', '방적 및 가공사 제조업', '합성고무 및 플라스틱 물질 제조업', '비료, 농약 및 살균, 살충제 제조업',  '마그네틱 및 광학 매체 제조업', '악기 제조업', '담배 제조업', '고무제품 제조업', '구조용 금속제품, 탱크 및 증기발생기 제조업', '인형,장난감 및 오락용품 제조업', '1차 비철금속 제조업', '펄프, 종이 및 판지 제조업', '그외 기타 제품 제조업', '가구 제조업']

  all_sector=[]
  for sector in (software,mechanic, automobile,electronic, medical, economy, chemistry, entertainment, science, food, clothes, transport, construction, wholesale_retail, other_service, other_manufacture):
    all_sector.append(sector)
  all_label = ['software','mechanic','automobile','electronic', 'medical', 'economy', 'chemistry', 'entertainment', 'science', 'food', 'clothes', 'transport', 'construction', 'wholesale_retail', 'other_service', 'other_manufacture']
  processed_label = []
  for labels in sector_label:
    sector_label=''
    for sector,label in zip(all_sector, all_label):
      if labels in sector:
        sector_label = label
    if sector_label=='':
      print(labels,"는 어느 섹터에도 속하지 않습니다. 우선은 기타 서비스업으로 분류합니다")
      sector_label = 'other_service'
    processed_label.append(sector_label)
  return processed_label


def get_dates(last_day, today):
  dates=[]
  lastday = datetime.datetime.strptime(str(last_day),'%Y-%m-%d %H:%M:%S')
  to = datetime.datetime.strptime(str(today),'%Y-%m-%d %H:%M:%S')
  # with two timestamp, return days between them. include last_day, not include today.
  delta = to-lastday
  for i in range(delta.days):
    day = lastday + datetime.timedelta(days=i)
    dates.append(day)

  return dates