#include <HX711.h>

#define thumb_data_pin A9
#define index_data_pin A7
#define middle_data_pin A5
#define ring_data_pin A3
#define pinky_data_pin A1

#define thumb_clock_pin A8
#define index_clock_pin A6
#define middle_clock_pin A4
#define ring_clock_pin A2
#define pinky_clock_pin A0

HX711 thumb, index, middle, ring, pinky;


void setup() {
  Serial.begin(9600);
  
  // Assigns pins to each sensor
  thumb.begin(thumb_data_pin, thumb_clock_pin);
  index.begin(index_data_pin, index_clock_pin);
  middle.begin(middle_data_pin, middle_clock_pin);
  ring.begin(ring_data_pin, ring_clock_pin);
  pinky.begin(pinky_data_pin, pinky_clock_pin);

  // Zeros the values of the sensors
  delay(2000);
  
  thumb.tare();
  index.tare();
  middle.tare();
  ring.tare();
  pinky.tare();
  
  delay(500);
}

void loop() {
  Serial.println((String) " a " + (long)thumb.get_units());
  Serial.println((String) " b " + (long)index.get_units());
  Serial.println((String) " c " + (long)middle.get_units());
  Serial.println((String) " d " + (long)ring.get_units());
  Serial.println((String) " e " + (long)pinky.get_units());
}
