#include <Servo.h>

/* ---------- SERVOS ---------- */
Servo thumbL, indexL, middleL, ringL, pinkyL, wristL;
Servo thumbR, indexR, middleR, ringR, pinkyR, wristR;

#define OPEN_ANGLE   160
#define CLOSE_ANGLE  20

/* ---------- LEFT BICEP ---------- */
#define EN_L   9
#define IN1_L  34
#define IN2_L  36
#define POT_L  A0

/* ---------- RIGHT BICEP ---------- */
#define EN_R   10
#define IN1_R  38
#define IN2_R  40
#define POT_R  A1

/* ---------- CONTROL ---------- */
#define PWM_KICK   255
#define PWM_FAST   230
#define PWM_SLOW   200
#define TOLERANCE    8
#define TIMEOUT_MS 6000

// RIGHT BICEP (unchanged)
#define BICEP_UP_POS_R    400
#define BICEP_DOWN_POS_R  600

// LEFT BICEP (reversed)
#define BICEP_UP_POS_L    600
#define BICEP_DOWN_POS_L  400

/* ---------- STATE ---------- */
bool leftActive = false, rightActive = false;
int leftTarget = 0, rightTarget = 0;

unsigned long leftStartTime = 0;
unsigned long rightStartTime = 0;

/* ---------- DEBUG ---------- */
unsigned long lastDebug = 0;
#define DEBUG_INTERVAL 200

/* ---------- MOTOR HELPERS ---------- */
void stopBicep(int EN, int IN1, int IN2) {
  analogWrite(EN, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void driveBicep(int EN, int IN1, int IN2, bool cw, int pwm) {
  digitalWrite(IN1, cw ? HIGH : LOW);
  digitalWrite(IN2, cw ? LOW : HIGH);
  analogWrite(EN, pwm);
}

/* ---------- UPDATE BICEPS ---------- */
void updateBiceps() {
  unsigned long now = millis();

  /* ===== LEFT ===== */
  if (leftActive) {
    int cur = analogRead(POT_L);
    int err = leftTarget - cur;
    int pwm = 0;

    if (abs(err) <= TOLERANCE) {
      stopBicep(EN_L, IN1_L, IN2_L);
      leftActive = false;
      Serial.println("[LEFT ] LOCKED");
    }
    else if (now - leftStartTime > TIMEOUT_MS) {
      stopBicep(EN_L, IN1_L, IN2_L);
      leftActive = false;
      Serial.println("[LEFT ] TIMEOUT");
    }
    else {
      int aerr = abs(err);
      if (aerr > 120)      pwm = PWM_KICK;
      else if (aerr > 50)  pwm = PWM_FAST;
      else                 pwm = PWM_SLOW;

      driveBicep(EN_L, IN1_L, IN2_L, err > 0, pwm);
    }

    if (now - lastDebug > DEBUG_INTERVAL) {
      Serial.print("[LEFT ] POT=");
      Serial.print(cur);
      Serial.print(" TARGET=");
      Serial.print(leftTarget);
      Serial.print(" PWM=");
      Serial.println(pwm);
    }
  }

  /* ===== RIGHT ===== */
  if (rightActive) {
    int cur = analogRead(POT_R);
    int err = rightTarget - cur;
    int pwm = 0;

    if (abs(err) <= TOLERANCE) {
      stopBicep(EN_R, IN1_R, IN2_R);
      rightActive = false;
      Serial.println("[RIGHT] LOCKED");
    }
    else if (now - rightStartTime > TIMEOUT_MS) {
      stopBicep(EN_R, IN1_R, IN2_R);
      rightActive = false;
      Serial.println("[RIGHT] TIMEOUT");
    }
    else {
      int aerr = abs(err);
      if (aerr > 120)      pwm = PWM_KICK;
      else if (aerr > 50)  pwm = PWM_FAST;
      else                 pwm = PWM_SLOW;

      driveBicep(EN_R, IN1_R, IN2_R, err > 0, pwm);
    }

    if (now - lastDebug > DEBUG_INTERVAL) {
      Serial.print("[RIGHT] POT=");
      Serial.print(cur);
      Serial.print(" TARGET=");
      Serial.print(rightTarget);
      Serial.print(" PWM=");
      Serial.println(pwm);
      lastDebug = now;
    }
  }
}

/* ---------- HAND FUNCTIONS ---------- */
void openLeftHand(){ thumbL.write(OPEN_ANGLE); indexL.write(OPEN_ANGLE); middleL.write(OPEN_ANGLE); ringL.write(OPEN_ANGLE); pinkyL.write(OPEN_ANGLE); }
void closeLeftHand(){ thumbL.write(CLOSE_ANGLE); indexL.write(CLOSE_ANGLE); middleL.write(CLOSE_ANGLE); ringL.write(CLOSE_ANGLE); pinkyL.write(CLOSE_ANGLE); }
void openRightHand(){ thumbR.write(OPEN_ANGLE); indexR.write(OPEN_ANGLE); middleR.write(OPEN_ANGLE); ringR.write(OPEN_ANGLE); pinkyR.write(OPEN_ANGLE); }
void closeRightHand(){ thumbR.write(CLOSE_ANGLE); indexR.write(CLOSE_ANGLE); middleR.write(CLOSE_ANGLE); ringR.write(CLOSE_ANGLE); pinkyR.write(CLOSE_ANGLE); }

/* ---------- COMMAND HANDLER ---------- */
void handleCommand(const String &cmd) {

  if (cmd == "LC") closeLeftHand();
  else if (cmd == "LO") openLeftHand();
  else if (cmd == "RC") closeRightHand();
  else if (cmd == "RO") openRightHand();

  else if (cmd.startsWith("WL"))
    wristL.write(constrain(cmd.substring(2).toInt(), 0, 180));
  else if (cmd.startsWith("WR"))
    wristR.write(constrain(cmd.substring(2).toInt(), 0, 180));

  // LEFT (reversed)
  else if (cmd == "LU") { leftTarget = BICEP_UP_POS_L; leftActive = true; leftStartTime = millis(); }
  else if (cmd == "LD") { leftTarget = BICEP_DOWN_POS_L; leftActive = true; leftStartTime = millis(); }

  // RIGHT (unchanged)
  else if (cmd == "RU") { rightTarget = BICEP_UP_POS_R; rightActive = true; rightStartTime = millis(); }
  else if (cmd == "RD") { rightTarget = BICEP_DOWN_POS_R; rightActive = true; rightStartTime = millis(); }

  else if (cmd == "BS") {
    leftActive = rightActive = false;
    stopBicep(EN_L, IN1_L, IN2_L);
    stopBicep(EN_R, IN1_R, IN2_R);
    Serial.println("BICEPS STOPPED");
  }
}

/* ---------- SERIAL ---------- */
void readCommand(Stream &s) {
  static String buf = "";
  while (s.available()) {
    char c = s.read();
    if (c == '\n' || c == '\r') {
      if (buf.length()) handleCommand(buf);
      buf = "";
    } else buf += c;
  }
}

/* ---------- SETUP ---------- */
void setup() {
  Serial.begin(115200);  // UPDATED TO 115200 for Python compatibility
  Serial1.begin(115200);

  thumbL.attach(50); indexL.attach(48); middleL.attach(46);
  ringL.attach(42); pinkyL.attach(44); wristL.attach(52);

  thumbR.attach(26); indexR.attach(28); middleR.attach(30);
  ringR.attach(32); pinkyR.attach(24); wristR.attach(22);

  pinMode(EN_L, OUTPUT); pinMode(IN1_L, OUTPUT); pinMode(IN2_L, OUTPUT);
  pinMode(EN_R, OUTPUT); pinMode(IN1_R, OUTPUT); pinMode(IN2_R, OUTPUT);

  Serial.println("=== BICEP + WRIST CONTROL READY ===");
}

/* ---------- LOOP ---------- */
void loop() {
  readCommand(Serial);
  readCommand(Serial1);
  updateBiceps();
}
