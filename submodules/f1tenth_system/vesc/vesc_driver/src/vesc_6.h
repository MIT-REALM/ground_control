#ifndef VESC_6_H_
#define VESC_6_H_

// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "kaitai/kaitaistruct.h"
#include <stdint.h>

#if KAITAI_STRUCT_VERSION < 9000L
#error "Incompatible Kaitai Struct C++/STL API: version 0.9 or later is required"
#endif

class vesc_6_t : public kaitai::kstruct {

public:

    vesc_6_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent = 0, vesc_6_t* p__root = 0);

private:
    void _read();
    void _clean_up();

public:
    ~vesc_6_t();

private:
    uint8_t m_packet_id;
    int16_t m_temp_fet_filtered;
    int16_t m_temp_motor_filtered;
    int32_t m_reset_avg_motor_current;
    int32_t m_reset_avg_input_current;
    int32_t m_reset_avg_id;
    int32_t m_reset_avg_iq;
    int16_t m_duty_cycle_now;
    int32_t m_rpm;
    int16_t m_input_voltage_filtered;
    int32_t m_amp_hours;
    int32_t m_amp_hours_charged;
    int32_t m_watt_hours;
    int32_t m_watt_hours_charged;
    int32_t m_tachometer_value;
    int32_t m_tachometer_abs_value;
    uint8_t m_fault;
    int16_t m_pid_pos_now;
    uint8_t m_controller_id;
    int16_t m_temp_mos1;
    int16_t m_temp_mos2;
    int16_t m_temp_mos3;
    int32_t m_reset_avg_vd;
    int32_t m_reset_avg_vq;
    vesc_6_t* m__root;
    kaitai::kstruct* m__parent;

public:
    uint8_t packet_id() const { return m_packet_id; }
    int16_t temp_fet_filtered() const { return m_temp_fet_filtered; }
    int16_t temp_motor_filtered() const { return m_temp_motor_filtered; }
    int32_t reset_avg_motor_current() const { return m_reset_avg_motor_current; }
    int32_t reset_avg_input_current() const { return m_reset_avg_input_current; }
    int32_t reset_avg_id() const { return m_reset_avg_id; }
    int32_t reset_avg_iq() const { return m_reset_avg_iq; }
    int16_t duty_cycle_now() const { return m_duty_cycle_now; }
    int32_t rpm() const { return m_rpm; }
    int16_t input_voltage_filtered() const { return m_input_voltage_filtered; }
    int32_t amp_hours() const { return m_amp_hours; }
    int32_t amp_hours_charged() const { return m_amp_hours_charged; }
    int32_t watt_hours() const { return m_watt_hours; }
    int32_t watt_hours_charged() const { return m_watt_hours_charged; }
    int32_t tachometer_value() const { return m_tachometer_value; }
    int32_t tachometer_abs_value() const { return m_tachometer_abs_value; }
    uint8_t fault() const { return m_fault; }
    int16_t pid_pos_now() const { return m_pid_pos_now; }
    uint8_t controller_id() const { return m_controller_id; }
    int16_t temp_mos1() const { return m_temp_mos1; }
    int16_t temp_mos2() const { return m_temp_mos2; }
    int16_t temp_mos3() const { return m_temp_mos3; }
    int32_t reset_avg_vd() const { return m_reset_avg_vd; }
    int32_t reset_avg_vq() const { return m_reset_avg_vq; }
    vesc_6_t* _root() const { return m__root; }
    kaitai::kstruct* _parent() const { return m__parent; }
};

#endif  // VESC_6_H_
