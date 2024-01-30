//Taehong Kim

#include <Arduino.h>
#include <MeMCore.h>
MeDCMotor motor1(M1);  
MeDCMotor motor2(M2);
MeLineFollower lineFollower(PORT_2); 
MeUltrasonicSensor ultrasonic(PORT_3);
unsigned long startMillis;  
unsigned long currentMillis;
const unsigned long period = 1000;
const byte ledPin = 13;   

int i = 0;

//turn 90 delay 800

void setup(){
  pinMode(A7, INPUT);//Start Code when buttonpressed
  while(analogRead(A7) != 0);
  Serial.begin(9600);
}


void loop(){
  if (ultrasonic.distanceCm() < 7){
    if(i==0){
      motor1.run(150);
      motor2.run(150);
      delay(570);
      i++;
      motor1.stop();
      motor2.stop();
      delay(100);
    }
    else if (i == 1){
      motor1.run(-150);
      motor2.run(-150);
      delay(1000);
      i++;
      motor1.stop();
      motor2.stop();
      delay(100);
    }
    else{
      motor1.run(150);
      motor2.run(150);
      delay(590);
      motor1.run(150);
      motor2.run(-150);
      delay(1400);
      motor1.run(150);
      motor2.run(150);
      delay(570);
      motor1.run(-150);
      motor2.run(-150);
      delay(1150);
      motor1.stop();
      motor2.stop();
      delay(100);
    }
    

  }
  else{
    i=0;
    motor1.run(-150);
    motor2.run(150);

  }
}
