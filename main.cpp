#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <random>
#include <iostream>

int MOTION = 1;

static std::default_random_engine engine(10);
static std::uniform_real_distribution<double> uniform(0, 1);


double gamma = 2.2;


class Vector {
public:
    double data[3];
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
    explicit Ray (Vector origin_, Vector direction_):
            origin(origin_), direction(direction_) {};

    Vector at_t(double t){
        return this -> origin + t * this -> direction;
    }

    Vector origin;
    Vector direction;
};

class Intersection {
public:
    Vector Point, N, color;
    double time;
    int found;
    int id;

    explicit Intersection(Vector P, Vector N_, Vector color_, int found_, int id_):
        Point(P), N(N_), color(color_), found(found_), id(id_) {};

    Intersection() {
        Point = Vector();
        N = Vector();
        color = Vector();
        found = id = 0;
        time = 0;
    }
};


class Geometry {
public:
    bool mirror = false, refraction = false, inverse_norm = false, is_light = false;
    int id = 0;
    Vector color = Vector(), motion = Vector(), position = Vector();

    virtual Intersection inter(Ray ray);

    Geometry at_time(double t, Geometry at) {
        at = *this;
        at.position = at.position + t * at.motion;
        return at;
    }

};
Intersection Geometry::inter(Ray ray) {return Intersection();}

class Sphere : public Geometry{
public:

    explicit Sphere (double r, Vector p, Vector color_, Vector motion_ = Vector()):
    radius(r){
        color = color_;
        motion = motion_;
        position = p;
    }
    void make_mirror(){
        mirror = true;
    }

    void make_light(){
        is_light = true;
    }


    void make_transparent(double n1 = 1.4){
        refraction = true;
        medium = n1;
    };
    void inverse(){
        inverse_norm = true;
    }

    Intersection inter(Ray ray);
    double radius, medium = 1;



};


Intersection Sphere::inter(Ray ray){
    Intersection I;
    I.id = id;
    I.color = color;





    Vector u = ray.direction;
    Vector O = ray.origin;
    Vector C = position;
    if (MOTION){
        C = C +  (motion * (((double) (rand() % 100)) / 12));
    }
    double R = radius;
    Vector dist = O - C;

    double gradient = pow(dot(u, dist), 2) - dist.norm2() + pow(R, 2);

    if (gradient < 0){
        // no intersection
        return I;
    } else {
        double t1 = dot(u, -1 * dist) - sqrt(gradient);
        double t2 = t1 + 2 * sqrt(gradient);

        if (t2 < 0) {
            return I;
        } else if (t1 > 0) {
            I.time = t1;
            I.Point = ray.at_t(t1);
            I.found = true;
            I.N = I.Point - this -> position;
            I.N.normalize();
            return I;
        } else {
            I.time = t2;
            I.Point = ray.at_t(t2);
            I.found = true;
            I.N = I.Point - this -> position;
            I.N.normalize();
            return I;
        }
    }
}

//class LightSource {
//public:
//    explicit LightSource (Vector position_, double intensity_):
//            position(position_), intensity(intensity_) {};
//
//    double intensity;
//    Vector position;
//
//};

//int ray_sphere_inter(Ray ray, Sphere sphere, double& time){
//
//    Vector u = ray.direction;
//    Vector O = ray.origin;
//    Vector C = sphere.position;
//    double R = sphere.radius;
//    Vector dist = O - C;
//
//    double gradient = pow(dot(u, dist), 2) - dist.norm2() + pow(R, 2);
//
//    if (gradient < 0){
//        // no intersection
//        return 0;
//    } else {
//        double t1 = dot(u, -1 * dist) - sqrt(gradient);
//        double t2 = t1 + 2 * sqrt(gradient);
//
//        if (t2 < 0) {
//            return 0;
//        } else if (t1 > 0) {
//            time = t1;
//            return 1;
//        } else {
//            time = t2;
//            return 1;
//        }
//    }
//}

class Scene {
public:
    double intensity = 3E7; // light sphere to fix point light issue for indirect lighting ;
    Sphere light;

    std::vector<Geometry*> geometries = {&light};

    explicit Scene(double light_radius, Vector light_position):
    light(light_radius, light_position, Vector()), geometries() {
        light.make_light();

    };


    void push_back(Geometry *object) {
        object -> id = geometries.size();
        geometries.push_back(object);
    }

    Intersection closest_inter(Ray ray, double& time);
    int visibility(Vector P, Vector light_p);
    Vector getColor(const Ray &ray, int ray_depth);

    bool indirect = false, antialiasing = false, motion = false;
};

Intersection Scene::closest_inter(Ray ray, double& time) {
//    if (ray.origin.data[0] != 0){
//        std::cout << "x:  " << ray.origin.data[0] << "y: " << ray.origin.data[1] << "z: " << ray.origin.data[2] << '\n';
//    }
    Intersection I;
    for (Geometry *object: geometries){
        Intersection res = Intersection();
        res = object -> inter(ray);
        if (res.found and (res.time < time or time < 0)) {
            I = res;
            time = res.time;
        }
    }
    return I;
}

int Scene::visibility(Vector P, Vector light_p){
    Vector direction = light_p - P;
    double d = direction.norm();

    direction.normalize();
    Ray light_ray = Ray(P, direction);

    double inter_time = -1;
    Intersection res = closest_inter(light_ray, inter_time);
    if (res.found) {
        return res.time > d - 10;
    } else {
        return true;
    }
}


Vector lambertian_scatter(Vector P, Vector albedo, Vector N, Scene scene) {
    Vector light_pos = scene.light.position;
    double intensity = scene.intensity;
    Vector omega = light_pos - P;
    double d = omega.norm();
    omega.normalize();
    N.normalize();

    Vector L = albedo * (intensity * scene.visibility(P + 1e-10 * N, light_pos) * fmax(dot(N, omega), 0)) / (4 * pow(M_PI,2) * pow(d, 2));
    //std::cout << L << '\n';
    return L;
}

Vector random_cos(Vector N) {
    //get a random vector for indirect lighting
    double r1 = uniform(engine), r2 = uniform(engine);
    double x = cos(2 * M_PI * r1) * pow(1 - r2, 0.5), y = sin(2 * M_PI * r1) * pow(1 - r2, 0.5);
    double z = sqrt(r2);
    Vector T1;

    int i = 0,j = 0;
    for (j = 1; j < 3; j++) {
        if (abs(N.data[j]) <= abs(N.data[i])) {
            i = j;
        }
    }
    if (i == 0) T1 = Vector(0., -N[2], N[1]);
    if (i == 1) T1 = Vector(-N[2], 0., N[0]);
    if (i == 2) T1 = Vector(-N[1], N[0], 0.);
    //normalize
    T1.normalize();
    Vector T2 = cross(N, T1);
    T2.normalize();
    Vector V = x * T1 + y * T2 + z * N;
    V.normalize();
    return V;
}

Vector Scene::getColor(const Ray& ray , int ray_depth) {
    if (ray_depth < 0) return Vector();

    double time = -1;
    Intersection I = closest_inter(ray, time);
    if (I.found) {
        if (geometries[I.id] -> mirror) {
            Ray reflected_ray = Ray(I.Point + 1e-10 * I.N, ray.direction - 2 * dot(ray.direction, I.N) * I.N);
            return getColor(reflected_ray,ray_depth - 1);
        } else if (geometries[I.id] -> refraction) {

            // for the inner hollow ball
            if (geometries[I.id] -> inverse_norm){ I.N = -1 * I.N;}
            double n1 = 1., n2 = 1.4;
            Vector prev_N = I.N;

            // when exiting sphere
            if (dot(ray.direction,I.N) > 0) {
                I.N = (-1.)*I.N;
                n1 = 1.4;
                n2 = 1.;
            }

            double k0 = pow(n1 - n2, 2.) / pow(n1 + n2, 2.);
            I.Point = I.Point - I.N*1e-10;


            double ang = dot(ray.direction,I.N);
            // test whether this is a totally internal reflection
            if (1. - pow(n1/n2, 2.) * (1 - pow(ang, 2.)) > 0) {
                Vector tangential = (n1/n2) * (ray.direction - ang*I.N);
                Vector normal = (-1.)* I.N * sqrt(1 - pow((n1/n2),2.) * (1 - pow(ang,2.)));

                Vector transmitted = tangential + normal;
                double prob = ((double) rand() / (RAND_MAX));
                if (prob < k0 + (1 - k0)*pow(1 - abs(dot(I.N,transmitted)),5.)) {
                    Ray reflected_ray = Ray(I.Point, ray.direction - (2*dot(ray.direction,prev_N)) * prev_N);
                    return getColor(reflected_ray, ray_depth - 1);
                } else {
                    Ray refracted_ray = Ray(I.Point, transmitted);
                    return getColor(refracted_ray, ray_depth - 1);
                }
            } else {
                Ray internal = Ray(I.Point, ray.direction - (2*dot(ray.direction,prev_N)) * prev_N);
                return getColor(internal, ray_depth - 1);
            }





//            double n1 = 1, n2 = 1.4;
//            double ang = dot(ray.direction, N);
//            double offset = -1e-10;
//
//
//            double k0 = pow(0.4, 2.)/pow(2.4, 2.);
//
//            if (ang > 0) {N = -1 * N; n2 = 1.0; n1 = 1.4; }
//            double n1dn2 = n1/n2;
//
//
//
//            Vector tangential = n1dn2 * (ray.direction - ang * N);
//            Vector nor = -1 * N * sqrt(1 - pow(n1dn2, 2) * (1 - pow(dot(ray.direction, N),2)));
//            Ray transmitted_ray = Ray(P - offset * N, tangential + nor);
//            return getColor(transmitted_ray, ray_depth - 1);
        } else if (geometries[I.id] -> is_light) {
            return Vector(255,255,255);
        } else{

            Vector L = lambertian_scatter(I.Point, I.color, I.N, *this);

            if (!indirect) {
                return L;
            }

            Ray incoming = Ray(I.Point + 1e-10 * I.N, random_cos(I.N));
            I.color.normalize();

            Vector temp = getColor(incoming, ray_depth - 1);
            Vector increment = Vector(temp[0] * I.color[0], temp[1] * I.color[1], temp[2] * I.color[2]);
            return L + increment;
        }
    }
};
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Vector get_normal(Vector P, Vector C){
    Vector distance = P - C;
    distance.normalize();
    return distance;
}





double correct(double x){
    x = fmin(255, x);
    x = x/255;
    x = pow(x, 1.0 / 2.2);
    x *= 255;
    return x;
}

void boxMuller(double stdev, double& x, double& y) {
    double r1 = uniform(engine);
    double r2 = uniform(engine);
    x = sqrt(-2 * log(r1)) * cos(2 * M_PI * r2) * stdev;
    y = sqrt(-2 * log(r1)) * sin(2 * M_PI * r2) * stdev;
    // x = 0, y = 0;
}


class TriangleIndices {
public:
    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
    };
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;  // indices within the uv coordinates array
    int ni, nj, nk;  // indices within the normals array
    int group;       // face group
};


class TriangleMesh : public Geometry {
public:
    ~TriangleMesh() {}
    TriangleMesh() {};

    TriangleMesh(Vector color_) {
        color = color_;
    }

    void readOBJ(const char* obj) {

        char matfile[255];
        char grp[255];

        FILE* f;
        f = fopen(obj, "r");
        int curGroup = -1;
        while (!feof(f)) {
            char line[255];
            if (!fgets(line, 255, f)) break;

            std::string linetrim(line);
            linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
            strcpy(line, linetrim.c_str());

            if (line[0] == 'u' && line[1] == 's') {
                sscanf(line, "usemtl %[^\n]\n", grp);
                curGroup++;
            }

            if (line[0] == 'v' && line[1] == ' ') {
                Vector vec;

                Vector col;
                if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
                    col[0] = std::min(1., std::max(0., col[0]));
                    col[1] = std::min(1., std::max(0., col[1]));
                    col[2] = std::min(1., std::max(0., col[2]));

                    vertices.push_back(vec);
                    vertexcolors.push_back(col);

                } else {
                    sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                    vertices.push_back(vec);
                }
            }
            if (line[0] == 'v' && line[1] == 'n') {
                Vector vec;
                sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                normals.push_back(vec);
            }
            if (line[0] == 'v' && line[1] == 't') {
                Vector vec;
                sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
                uvs.push_back(vec);
            }
            if (line[0] == 'f') {
                TriangleIndices t;
                int i0, i1, i2, i3;
                int j0, j1, j2, j3;
                int k0, k1, k2, k3;
                int nn;
                t.group = curGroup;

                char* consumedline = line + 1;
                int offset;

                nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
                if (nn == 9) {
                    if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
                    if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
                    if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
                    if (j0 < 0) t.uvi = uvs.size() + j0; else	t.uvi = j0 - 1;
                    if (j1 < 0) t.uvj = uvs.size() + j1; else	t.uvj = j1 - 1;
                    if (j2 < 0) t.uvk = uvs.size() + j2; else	t.uvk = j2 - 1;
                    if (k0 < 0) t.ni = normals.size() + k0; else	t.ni = k0 - 1;
                    if (k1 < 0) t.nj = normals.size() + k1; else	t.nj = k1 - 1;
                    if (k2 < 0) t.nk = normals.size() + k2; else	t.nk = k2 - 1;
                    indices.push_back(t);
                } else {
                    nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
                    if (nn == 6) {
                        if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
                        if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
                        if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
                        if (j0 < 0) t.uvi = uvs.size() + j0; else	t.uvi = j0 - 1;
                        if (j1 < 0) t.uvj = uvs.size() + j1; else	t.uvj = j1 - 1;
                        if (j2 < 0) t.uvk = uvs.size() + j2; else	t.uvk = j2 - 1;
                        indices.push_back(t);
                    } else {
                        nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
                        if (nn == 3) {
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
                            indices.push_back(t);
                        } else {
                            nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else	t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else	t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else	t.vtxk = i2 - 1;
                            if (k0 < 0) t.ni = normals.size() + k0; else	t.ni = k0 - 1;
                            if (k1 < 0) t.nj = normals.size() + k1; else	t.nj = k1 - 1;
                            if (k2 < 0) t.nk = normals.size() + k2; else	t.nk = k2 - 1;
                            indices.push_back(t);
                        }
                    }
                }

                consumedline = consumedline + offset;

                while (true) {
                    if (consumedline[0] == '\n') break;
                    if (consumedline[0] == '\0') break;
                    nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
                    TriangleIndices t2;
                    t2.group = curGroup;
                    if (nn == 3) {
                        if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
                        if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
                        if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
                        if (j0 < 0) t2.uvi = uvs.size() + j0; else	t2.uvi = j0 - 1;
                        if (j2 < 0) t2.uvj = uvs.size() + j2; else	t2.uvj = j2 - 1;
                        if (j3 < 0) t2.uvk = uvs.size() + j3; else	t2.uvk = j3 - 1;
                        if (k0 < 0) t2.ni = normals.size() + k0; else	t2.ni = k0 - 1;
                        if (k2 < 0) t2.nj = normals.size() + k2; else	t2.nj = k2 - 1;
                        if (k3 < 0) t2.nk = normals.size() + k3; else	t2.nk = k3 - 1;
                        indices.push_back(t2);
                        consumedline = consumedline + offset;
                        i2 = i3;
                        j2 = j3;
                        k2 = k3;
                    } else {
                        nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
                        if (nn == 2) {
                            if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
                            if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
                            if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
                            if (j0 < 0) t2.uvi = uvs.size() + j0; else	t2.uvi = j0 - 1;
                            if (j2 < 0) t2.uvj = uvs.size() + j2; else	t2.uvj = j2 - 1;
                            if (j3 < 0) t2.uvk = uvs.size() + j3; else	t2.uvk = j3 - 1;
                            consumedline = consumedline + offset;
                            i2 = i3;
                            j2 = j3;
                            indices.push_back(t2);
                        } else {
                            nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
                            if (nn == 2) {
                                if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
                                if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
                                if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
                                if (k0 < 0) t2.ni = normals.size() + k0; else	t2.ni = k0 - 1;
                                if (k2 < 0) t2.nj = normals.size() + k2; else	t2.nj = k2 - 1;
                                if (k3 < 0) t2.nk = normals.size() + k3; else	t2.nk = k3 - 1;
                                consumedline = consumedline + offset;
                                i2 = i3;
                                k2 = k3;
                                indices.push_back(t2);
                            } else {
                                nn = sscanf(consumedline, "%u%n", &i3, &offset);
                                if (nn == 1) {
                                    if (i0 < 0) t2.vtxi = vertices.size() + i0; else	t2.vtxi = i0 - 1;
                                    if (i2 < 0) t2.vtxj = vertices.size() + i2; else	t2.vtxj = i2 - 1;
                                    if (i3 < 0) t2.vtxk = vertices.size() + i3; else	t2.vtxk = i3 - 1;
                                    consumedline = consumedline + offset;
                                    i2 = i3;
                                    indices.push_back(t2);
                                } else {
                                    consumedline = consumedline + 1;
                                }
                            }
                        }
                    }
                }

            }

        }
        fclose(f);

    }

    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;

};


int main() {
    // We use 512x512 pixels with angle 60 starting at origin (0,0,55);
    int W = 512;
    int H = 512;
    Vector Q = Vector(0,0,55);
    int alpha = 60 * M_PI / 180;
    double d = double(W) / (2 * tan(double(alpha) / 2));
    int run_config = 5;
    int PER_PIXEL = 10;
    const char* save_name = "light_radius_10pixels.png";



    double light_radius = 1;
    Vector light_position = Vector(-10, 20, 40);
    Scene scene = Scene(light_radius, light_position);
    //~~~~~~~~~~~~~ Basic ray tracing with shadows SCENE 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if (run_config == 0) {
        // ~10 seconds
        Sphere* center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));

        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = false;
        scene.antialiasing = false;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);
    } else if (run_config == 1){
        // ~2 seconds for 1 pixel, ~ 12 seconds for 10 pixels, ~1m29s for 100 pixels
        // ~~~~~~~~~~~~~~~~ reflection, refraction and hollow spheres ~~~~~~~~~~~~~~~~~
        Sphere* center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));
        center_sphere -> make_transparent();
        Sphere* left_sphere = new Sphere(9, Vector(-20, 0, 0), Vector(.2, .2, .2));
        left_sphere -> make_mirror();
        Sphere* right_sphere = new Sphere(9, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere -> make_transparent();
        Sphere* right_sphere2 = new Sphere(8.999, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere2 -> make_transparent();
        right_sphere2 -> inverse();


        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = false;
        scene.antialiasing = false;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(left_sphere);
        scene.push_back(right_sphere);
        scene.push_back(right_sphere2);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);

    } else if (run_config == 2){
        // ~32 seconds for 10 pixels, ~5m10s for 100 pixels
        // ~~~~~~~~~~~~~~~~ indirect lighting  ~~~~~~~~~~~~~~~~~
        Sphere* center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));
        center_sphere -> make_transparent();
        Sphere* left_sphere = new Sphere(9, Vector(-20, 0, 0), Vector(.2, .2, .2));
        left_sphere -> make_mirror();
        Sphere* right_sphere = new Sphere(9, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere -> make_transparent();
        Sphere* right_sphere2 = new Sphere(8.999, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere2 -> make_transparent();
        right_sphere2 -> inverse();


        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = true;
        scene.antialiasing = false;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(left_sphere);
        scene.push_back(right_sphere);
        scene.push_back(right_sphere2);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);



    } else if (run_config == 3){
        // ~9s for 1 pixel, ~38s for 10 pixels, ~ 5m45s for 100 pixels, 50 mins
        // ~~~~~~~~~~~~~~~~ indirect lighting  ~~~~~~~~~~~~~~~~~
        Sphere* center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));
        center_sphere -> make_transparent();
        Sphere* left_sphere = new Sphere(9, Vector(-20, 0, 0), Vector(.2, .2, .2));
        left_sphere -> make_mirror();
        Sphere* right_sphere = new Sphere(9, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere -> make_transparent();
        Sphere* right_sphere2 = new Sphere(8.999, Vector(20, 0, 0), Vector(.2, .2, .2));
        right_sphere2 -> make_transparent();
        right_sphere2 -> inverse();


        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = true;
        scene.antialiasing = true;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(left_sphere);
        scene.push_back(right_sphere);
        scene.push_back(right_sphere2);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);

    } else if (run_config == 4) {
        // ~10 seconds
        Sphere *center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2), Vector(1,0,0));

        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = false;
        scene.antialiasing = false;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);

    } else if (run_config == 5) {
        Sphere* center_sphere = new Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));

        Sphere* green_sphere = new Sphere(940, Vector(0, 0, -1000), Vector(.1, .4, .1));
        Sphere* red_sphere = new Sphere(940, Vector(0, 1000, 0), Vector(.4, .1, .1));
        Sphere* pink_sphere = new Sphere(940, Vector(0, 0, 1000), Vector(.4, .1, .4));
        Sphere* blue_sphere = new Sphere(990, Vector(0, -1000, 0), Vector(.1, .1, .4));
        Sphere* yellow_sphere = new Sphere(940, Vector(1000, 0, 0), Vector(.4, .4, .1));
        Sphere* light_blue_sphere = new Sphere(940, Vector(-1000, 0, 0), Vector(.4, .4, .9));


        scene.indirect = false;
        scene.antialiasing = false;
        scene.motion = false;

        scene.push_back(center_sphere);
        scene.push_back(green_sphere);
        scene.push_back(red_sphere);
        scene.push_back(pink_sphere);
        scene.push_back(blue_sphere);
        scene.push_back(light_blue_sphere);
        scene.push_back(yellow_sphere);

        scene.light.radius = 1000;
        scene.light.position = Vector(10, 50, 40);
    }



//    Sphere left_sphere = Sphere(9, Vector(-20, 0, 0), Vector(.2, .2, .2), Vector(0,1,0));
//    //left_sphere.make_mirror();
//    Sphere center_sphere = Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));
//    center_sphere.make_transparent();
//    Sphere right_sphere = Sphere(9, Vector(20, 0, 0), Vector(.2, .2, .2));
//    right_sphere.make_transparent();
//    Sphere right_sphere2 = Sphere(8.999, Vector(20, 0, 0), Vector(.2, .2, .2));
//    right_sphere2.make_transparent();
//    right_sphere2.inverse();
//    Sphere green_sphere = Sphere(940, Vector(0,0,-1000), Vector(.1, .4, .1));
//    Sphere red_sphere = Sphere(940, Vector(0,1000,0), Vector(.4, .1, .1));
//    Sphere pink_sphere = Sphere(940, Vector(0,0,1000), Vector(.4, .1, .4));
//    Sphere blue_sphere = Sphere(990, Vector(0,-1000,0), Vector(.1, .1, .4));
//    Sphere yellow_sphere = Sphere(940, Vector(1000, 0, 0), Vector(.4,.4,.1));
//    Sphere light_blue_sphere = Sphere(940, Vector(-1000, 0, 0), Vector(.4,.4,.9));
//
//    double light_radius = 1;
//    Vector light_position = Vector(-10, 20, 40);
//    Scene scene = Scene(light_radius, light_position);
//    scene.indirect = true;
//    scene.antialiasing = false;
//
//
//    scene.push_back(&left_sphere);
//    scene.push_back(&center_sphere);
//    scene.push_back(&right_sphere);
//    scene.push_back(&right_sphere2);
//
//    scene.push_back(&green_sphere);
//    scene.push_back(&red_sphere);
//    scene.push_back(&pink_sphere);
//    scene.push_back(&blue_sphere);
//    scene.push_back(&light_blue_sphere);
//    scene.push_back(&yellow_sphere);

    // double max_color = 0;

    std::vector<unsigned char> image(W * H * 3, 0);
    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < H; i++) {

        std::cout << i << '\n';
        for (int j = 0; j < W; j++) {

            // build ray for given pixel
            double x_val = Q.data[0] + double(j) - double(W) / 2 + 0.5;
            double y_val = Q.data[1] + (double(H) - double(i) - 1) - double(H) / 2 + 0.5;
            double z_val = Q.data[2] - d;
            Vector direction = Vector(x_val, y_val, z_val);
            //direction.normalize();
            Ray ray = Ray(Q, direction);

            Vector L;
            for (int k = 0; k < PER_PIXEL; k++){
                if (scene.antialiasing) {

                    double a = 0, b = 0;
                    boxMuller(.35, a, b);

                    Vector new_direction = Vector(a, b, 0) + ray.direction;
                    new_direction.normalize();
                    Ray temp_ray = Ray(ray.origin, new_direction);
                    L = L + scene.getColor(temp_ray,5);
                } else {
                    ray.direction.normalize();
                    L = L + scene.getColor(ray,5);
                }
            }
            L = L / PER_PIXEL;

            image[(i * W + j) * 3 + 0] = correct(L.data[0]);
            image[(i * W + j) * 3 + 1] = correct(L.data[1]);
            image[(i * W + j) * 3 + 2] = correct(L.data[2]);

//            Vector P, N;
//            if (scene.closest_inter(ray, ray_inter_time, intersected, P, N)) {


//                Vector L = lambertian_scatter(P, intersected.color, N, scene);
//                //std::cout << L.data[1]<< '\n';
//                image[(i * W + j) * 3 + 0] = correct(L.data[0]);
//                image[(i * W + j) * 3 + 1] = correct(L.data[1]);
//                image[(i * W + j) * 3 + 2] = correct(L.data[2]);

//                double constant1 = fmax(dot(normal, omega), 0);
//
//                Vector slight_modif = (intersected.color / M_PI) * fmax(dot(normal, omega), 0);
//                slight_modif = slight_modif *
//                image[(i * W + j) * 3 + 0] = intersected.color.data[0] * 255;
//                image[(i * W + j) * 3 + 1] = intersected.color.data[1] * 255;
////                image[(i * W + j) * 3 + 2] = intersected.color.data[2] * 255;
//                image[(i * W + j) * 3 + 0] = slight_modif.data[0] * 255;
//                image[(i * W + j) * 3 + 1] = slight_modif.data[1] * 255;
//                image[(i * W + j) * 3 + 2] = slight_modif.data[2] * 255;

            // max_color = fmax(fmax(fmax(max_color, L.data[0]), L.data[1]), L.data[2]);
//            } else {
//                image[(i * W + j) * 3 + 0] = 0;
//                image[(i * W + j) * 3 + 1] = 0;
//                image[(i * W + j) * 3 + 2] = 0;
//            }
        }
    }
    stbi_write_png(save_name, W, H, 3, &image[0], 0);

    for (Geometry* g: scene.geometries){
        delete g;
    }
    return 0;
}

