package leetcode;


import java.util.*;

public class Main {
    // Map的value值降序排序
    public static <K, V extends Comparable<? super V>> Map<K, V> sortDescend(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                int compare = (o1.getValue()).compareTo(o2.getValue());
                return -compare;
            }
        });

        Map<K, V> returnMap = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            returnMap.put(entry.getKey(), entry.getValue());
        }
        return returnMap;
    }

    // Map的value值升序排序
    public static <K, V extends Comparable<? super V>> Map<K, V> sortAscend(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new ArrayList<Map.Entry<K, V>>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                int compare = (o1.getValue()).compareTo(o2.getValue());
                return compare;
            }
        });

        Map<K, V> returnMap = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            returnMap.put(entry.getKey(), entry.getValue());
        }
        return returnMap;
    }

    public static void main(String[] args) {
    }

    private static Map<Integer, Room> map = new HashMap<>();

    public static boolean addRoom(int id, int area, int price, int rooms, int[] address) {
        if (map.containsKey(id)) {
            map.put(id, new Room(area, price, rooms, address));
            return false;
        } else {
            map.put(id, new Room(area, price, rooms, address));
            return true;
        }
    }

    private static boolean deleteRoom(int id) {
        if (map.containsKey(id)) {
            map.remove(id);
            return true;
        } else {
            return false;
        }
    }

    private static int[] queryRoom(int area, int price, int rooms, int[] address, int[][] orderBy) {
        List<Room> filterRoom = new ArrayList<>();
        for (Map.Entry<Integer, Room> integerRoomEntry : map.entrySet()) {
            Room value = integerRoomEntry.getValue();
            if (value.area >= area && value.price <= price && value.rooms == rooms) {
                value.setId(integerRoomEntry.getKey());
                filterRoom.add(value);
            }
        }
        return filterRoom.stream().sorted(new Comparator<Room>() {
            @Override
            public int compare(Room o1, Room o2) {
                int res = 0;
                for (int[] ints : orderBy) {
                    if (res != 0) {
                        break;
                    }
                    int rule = ints[0];
                    int upDown = ints[1];
                    switch (rule) {
                        case 1:
                            res = upDown == 1 ? o1.area - o2.area : o2.area - o1.area;
                            break;
                        case 2:
                            res = upDown == 1 ? o1.price - o2.price : o2.price - o1.price;
                            break;
                        case 3:
                            int way1 = cal(o1.address, address);
                            int way2 = cal(o2.address, address);
                            res = upDown == 1 ? way1 - way2 : way2 - way1;
                            break;
                    }
                }
                return res == 0 ? o1.id - o2.id : o2.id - o1.id;
            }
        }).mapToInt(room -> room.id).toArray();
    }

    public static int cal(int[] point, int[] desPoint) {
        return Math.abs(point[0] - desPoint[0]) + Math.abs(point[1] - desPoint[1]);
    }
}

class Room {
    int id;
    int area;
    int price;
    int rooms;

    public void setId(int id) {
        this.id = id;
    }

    int[] address;

    public Room(int area, int price, int rooms, int[] address) {
        this.area = area;
        this.price = price;
        this.rooms = rooms;
        this.address = address;
    }
}