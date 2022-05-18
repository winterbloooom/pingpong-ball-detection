"""
순서
1. 실측을 통한 obj point , img point 설정
2. intrinsic, distort coefficients 정보 불러오기
3. undistort
4. obj point -> homogeneous 좌표계로 변환
5. solvePnP, drawFrameAxes로 좌표축 확인, 0,0,0 좌표로 그려져야 맞는것
6. 

"""