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

int speed = 170;

//turn 90 delay 800

void setup(){
  pinMode(A7, INPUT);//Start Code when buttonpressed
  while(analogRead(A7) != 0);
  Serial.begin(9600);
  
}


void loop(){
  if (ultrasonic.distanceCm() < 7){
    
    motor1.run(-200);
    motor2.run(-200);
    delay(400);
    while(lineFollower.readSensor1()==1 && lineFollower.readSensor2()==1){
      motor1.run(-80);
      motor2.run(220);
    }
    motor1.run(-200);
    motor2.run(90);
    delay(800);

  }
  else{
    lineFollow();
  }
  
  Serial.print("Distance: ");
  Serial.print(ultrasonic.distanceCm());
  Serial.println(" cm");
  delay(100);
}

void lineFollow(){
  int sensorState = lineFollower.readSensors();


  switch(sensorState)
  {
    case S1_IN_S2_IN:
      moveForward();
      if (5000 <= (currentMillis - startMillis)){
        TurnLeft();
      }
      break;
    case S1_IN_S2_OUT:
      TurnLeft();
      currentMillis = millis();
      break;
    case S1_OUT_S2_IN:
      TurnRight();
      currentMillis = millis();
      break;
    case S1_OUT_S2_OUT:
      moveBackward();
      break;
  }
}




void moveForward()
{
  motor1.run(-speed);
  motor2.run(speed);
}
void moveBackward()
{
  motor1.run(100);
  motor2.run(-100);
}
void TurnLeft()
{
  motor1.run(-speed/3); // Turn left
  motor2.run(speed);
}
void TurnRight()
{
  motor1.run(-speed); // Turn right
  motor2.run(speed/3);
}
