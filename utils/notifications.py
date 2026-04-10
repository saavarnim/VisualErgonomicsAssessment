from plyer import notification
import time

class PostureNotifier:
    def __init__(self, cooldown_seconds=60):
        """
        Manages desktop notifications with a cooldown to avoid spamming the user.
        """
        self.cooldown_seconds = cooldown_seconds
        self.last_notification_time = 0

    def notify_bad_posture(self, current_state):
        """
        Triggers a Windows notification if the cooldown has passed.
        """
        current_time = time.time()
        
        if current_state == "Fatigued":
            if current_time - self.last_notification_time > self.cooldown_seconds:
                try:
                    notification.notify(
                        title="Posture Alert!",
                        message="You have been in a poor posture for too long. Please sit up straight and relax your shoulders.",
                        app_name="Visual Ergonomics",
                        timeout=10
                    )
                    self.last_notification_time = current_time
                    print(">>> Desktop Notification Sent")
                except Exception as e:
                    print(f"Error sending notification: {e}")
