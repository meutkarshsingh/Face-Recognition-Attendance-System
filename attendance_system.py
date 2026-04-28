"""Attendance system for marking attendance based on face recognition"""

import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import ATTENDANCE_DIR, LOGS_DIR, TIMEZONE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "attendance_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Manages attendance records"""

    def __init__(self):
        self.attendance_records = {}  # person_id -> list of attendance entries
        self.load_today_attendance()

    def get_attendance_file(self, date: Optional[datetime] = None) -> Path:
        """
        Get attendance file path for a specific date
        
        Args:
            date: Date to get file for (default: today)
            
        Returns:
            Path to attendance file
        """
        if date is None:
            date = datetime.now()
        
        filename = f"attendance_{date.strftime('%Y-%m-%d')}.csv"
        return ATTENDANCE_DIR / filename

    def load_today_attendance(self):
        """Load today's attendance records from file"""
        attendance_file = self.get_attendance_file()
        
        if not attendance_file.exists():
            logger.info("No attendance file for today")
            return
        
        try:
            with open(attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    person_id = row['person_id']
                    if person_id not in self.attendance_records:
                        self.attendance_records[person_id] = []
                    self.attendance_records[person_id].append(row)
            
            logger.info(f"Loaded {len(self.attendance_records)} person records")
        except Exception as e:
            logger.error(f"Error loading attendance: {e}")

    def mark_attendance(self, person_id: str, person_name: str, 
                       confidence: float, location: Dict) -> bool:
        """
        Mark attendance for a person
        
        Args:
            person_id: Person's ID
            person_name: Person's name
            confidence: Recognition confidence
            location: Face location dict
            
        Returns:
            True if attendance marked successfully
        """
        try:
            timestamp = datetime.now()
            
            entry = {
                'person_id': person_id,
                'person_name': person_name,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence': f"{confidence:.4f}",
                'face_location': str(location)
            }
            
            if person_id not in self.attendance_records:
                self.attendance_records[person_id] = []
            
            self.attendance_records[person_id].append(entry)
            
            logger.info(f"Marked attendance for {person_name} (ID: {person_id})")
            return True
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False

    def save_attendance(self) -> bool:
        """
        Save attendance records to file
        
        Returns:
            True if save successful
        """
        try:
            attendance_file = self.get_attendance_file()
            
            if not self.attendance_records:
                logger.info("No attendance records to save")
                return True
            
            # Flatten all records
            all_records = []
            for person_records in self.attendance_records.values():
                all_records.extend(person_records)
            
            # Write to CSV
            fieldnames = ['person_id', 'person_name', 'timestamp', 'confidence', 'face_location']
            
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)
            
            logger.info(f"Saved {len(all_records)} attendance records to {attendance_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving attendance: {e}")
            return False

    def get_attendance_summary(self) -> Dict:
        """
        Get summary of today's attendance
        
        Returns:
            Dictionary with attendance summary
        """
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_persons': len(self.attendance_records),
            'total_entries': sum(len(entries) for entries in self.attendance_records.values()),
            'persons': {}
        }
        
        for person_id, entries in self.attendance_records.items():
            if entries:
                latest_entry = entries[-1]
                summary['persons'][person_id] = {
                    'name': latest_entry['person_name'],
                    'entry_count': len(entries),
                    'first_entry': entries[0]['timestamp'],
                    'last_entry': latest_entry['timestamp'],
                    'avg_confidence': sum(float(e['confidence']) for e in entries) / len(entries)
                }
        
        return summary

    def is_person_present_today(self, person_id: str) -> bool:
        """
        Check if a person has been marked present today
        
        Args:
            person_id: Person's ID
            
        Returns:
            True if person is marked present today
        """
        return person_id in self.attendance_records and len(self.attendance_records[person_id]) > 0

    def get_person_entries_today(self, person_id: str) -> List[Dict]:
        """
        Get all attendance entries for a person today
        
        Args:
            person_id: Person's ID
            
        Returns:
            List of attendance entries
        """
        return self.attendance_records.get(str(person_id), [])
