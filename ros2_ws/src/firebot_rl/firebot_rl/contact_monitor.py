import math

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool, String


class ContactMonitor(Node):
    """Monitors /marble_hd2/contact and separates wall vs ground contacts.

    Publishes:
        /marble_hd2/wall_contact   (Bool)   — True while a wall/obstacle hit is active
        /marble_hd2/ground_contact (String) — name of the tile/room the tracks are on
    """

    def __init__(self):
        super().__init__('contact_monitor')

        # State tracking — only log on transitions
        self.in_wall_contact = False
        self.current_ground = ''

        # Subscriber
        self.create_subscription(
            Contacts,
            '/marble_hd2/contact',
            self.contact_callback,
            10,
        )

        # Publishers
        self.wall_pub = self.create_publisher(Bool, '/marble_hd2/wall_contact', 10)
        self.ground_pub = self.create_publisher(String, '/marble_hd2/ground_contact', 10)

        self.get_logger().info(
            'Contact monitor started — publishing wall_contact and ground_contact'
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    def contact_callback(self, msg: Contacts):
        wall_contacts = []
        ground_names = set()

        for contact in msg.contacts:
            is_wall = False
            for normal in contact.normals:
                if abs(normal.z) < 0.7:           # horizontal normal = wall
                    is_wall = True
                    break

            if is_wall:
                wall_contacts.append(contact)
            else:
                # Ground contact — extract the tile/room model name
                # Entity names look like "staging_area::base::collision"
                # We want the first component ("staging_area")
                for name in (contact.collision2.name, contact.collision1.name):
                    model = name.split('::')[0]
                    if model and model != 'marble_hd2':
                        ground_names.add(model)

        # ---- Wall contact topic ----
        has_wall = len(wall_contacts) > 0
        wall_msg = Bool()
        wall_msg.data = has_wall
        self.wall_pub.publish(wall_msg)

        if has_wall and not self.in_wall_contact:
            self.in_wall_contact = True
            self._log_wall_details(wall_contacts)
        elif not has_wall and self.in_wall_contact:
            self.in_wall_contact = False
            self.get_logger().info('Wall contact cleared.')

        # ---- Ground contact topic ----
        ground_str = ', '.join(sorted(ground_names)) if ground_names else ''
        ground_msg = String()
        ground_msg.data = ground_str
        self.ground_pub.publish(ground_msg)

        if ground_str and ground_str != self.current_ground:
            self.current_ground = ground_str
            self.get_logger().info(f'Ground: {ground_str}')
        elif not ground_str and self.current_ground:
            self.current_ground = ''

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_wall_details(self, contacts):
        """Log a summary of wall-contact event."""
        entities = set()
        total_depth = 0.0
        depth_count = 0
        total_force_sq = 0.0

        for contact in contacts:
            entities.add(contact.collision1.name)
            entities.add(contact.collision2.name)

            for d in contact.depths:
                total_depth += d
                depth_count += 1

            for w in contact.wrenches:
                f = w.body_1_wrench.force
                total_force_sq += f.x ** 2 + f.y ** 2 + f.z ** 2

        avg_depth = total_depth / depth_count if depth_count else 0.0
        force_mag = math.sqrt(total_force_sq)

        self.get_logger().warn(
            f'WALL COLLISION | '
            f'entities: {entities} | '
            f'avg depth: {avg_depth:.4f} | '
            f'force magnitude: {force_mag:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ContactMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
