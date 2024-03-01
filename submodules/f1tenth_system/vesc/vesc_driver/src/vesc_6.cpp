// This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

#include "vesc_6.h"

vesc_6_t::vesc_6_t(kaitai::kstream* p__io, kaitai::kstruct* p__parent, vesc_6_t* p__root) : kaitai::kstruct(p__io) {
    m__parent = p__parent;
    m__root = this;

    try {
        _read();
    } catch(...) {
        _clean_up();
        throw;
    }
}

void vesc_6_t::_read() {
    m_packet_id = m__io->read_u1();
    m_temp_fet_filtered = m__io->read_s2be();
    m_temp_motor_filtered = m__io->read_s2be();
    m_reset_avg_motor_current = m__io->read_s4be();
    m_reset_avg_input_current = m__io->read_s4be();
    m_reset_avg_id = m__io->read_s4be();
    m_reset_avg_iq = m__io->read_s4be();
    m_duty_cycle_now = m__io->read_s2be();
    m_rpm = m__io->read_s4be();
    m_input_voltage_filtered = m__io->read_s2be();
    m_amp_hours = m__io->read_s4be();
    m_amp_hours_charged = m__io->read_s4be();
    m_watt_hours = m__io->read_s4be();
    m_watt_hours_charged = m__io->read_s4be();
    m_tachometer_value = m__io->read_s4be();
    m_tachometer_abs_value = m__io->read_s4be();
    m_fault = m__io->read_u1();
    m_pid_pos_now = m__io->read_s2be();
    m_controller_id = m__io->read_u1();
    m_temp_mos1 = m__io->read_s2be();
    m_temp_mos2 = m__io->read_s2be();
    m_temp_mos3 = m__io->read_s2be();
    m_reset_avg_vd = m__io->read_s4be();
    m_reset_avg_vq = m__io->read_s4be();
}

vesc_6_t::~vesc_6_t() {
    _clean_up();
}

void vesc_6_t::_clean_up() {
}
